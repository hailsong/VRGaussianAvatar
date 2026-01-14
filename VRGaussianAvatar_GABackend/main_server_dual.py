import sys
import os
import io
import asyncio
import torch
torch._dynamo.config.disable = True

import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from PIL import Image
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf
import cv2

from time import perf_counter
from typing import List

# --- Configuration ---
class CONFIG:
    LHM_APP_PATH   = "./LHM"
    TARGET_IMAGE_PATH = "./LHM/train_data/example_imgs/p_male_0_input.jpg"
    MODEL_NAME     = "LHM-500M-HF"
    RENDER_WIDTH   = 512
    RENDER_HEIGHT  = 512
    IPD            = 0.055
    FOV_DEGREES    = 65.0
    JPEG_QUALITY   = 80 

sys.path.insert(0, os.path.abspath(CONFIG.LHM_APP_PATH))
MODULE_PATH = os.path.abspath(os.path.join(CONFIG.LHM_APP_PATH, "engine/pose_estimation"))
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

# --- LHM / Utils ---
from accelerate import PartialState
from LHM.utils.model_download_utils import AutoModelQuery
from LHM.utils.hf_hub import wrap_model_hub
from LHM.models import model_dict
from LHM.utils.model_card import MODEL_CONFIG
from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from LHM.utils.face_detector import VGGHeadDetector
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    resize_image_keepaspect_np,
)

try:
    from engine.SegmentAPI.SAM import SAM2Seg
except ImportError:
    from rembg import remove
    SAM2Seg = None
    print("SAM2 not found, using rembg.")

g = {
    "lhm": None, "gs_model_list": None, "query_points": None,
    "smplx_params": None, "bg_color": None, "transform_mat_neutral_pose": None
}

# -----------------------------
# Helpers
# -----------------------------
def parse_configs_local():
    cli_cfg, cfg = OmegaConf.create(), OmegaConf.create()
    query_model = AutoModelQuery()
    model_path = query_model.query(CONFIG.MODEL_NAME)
    cli_cfg.model_name = model_path
    model_params_key = CONFIG.MODEL_NAME.split('-')[1]
    model_config_path = MODEL_CONFIG.get(model_params_key)
    if model_config_path:
        relative_config_path = model_config_path.replace("LHM/", "", 1)
        cfg_train = OmegaConf.load(relative_config_path)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.src_head_size = cfg_train.dataset.get('src_head_size', 112)
        cfg.render_size = cfg_train.dataset.render_image.high
    cfg.merge_with(cli_cfg)
    return cfg

def build_model_local(cfg):
    hf_model_cls = wrap_model_hub(model_dict["human_lrm_sapdino_bh_sd3_5"])
    model = hf_model_cls.from_pretrained(cfg.model_name)
    return model

def get_bbox(mask):
    h, w = mask.shape
    y, x = np.where(mask > 128)
    if len(x) == 0 or len(y) == 0:
        return Bbox([0, 0, w, h])
    return Bbox([x.min(), y.min(), x.max(), y.max()]).scale(1.1, width=w, height=h)

def infer_preprocess_image_local(rgb_path, mask):
    cfg = parse_configs_local()
    rgb_pil = Image.open(rgb_path).convert("RGB")
    rgb = np.array(rgb_pil)
    bbox = get_bbox(mask)
    x0, y0, x1, y1 = bbox.get_box()
    rgb = rgb[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]

    h, w, _ = rgb.shape
    aspect_standard = 5.0 / 3.0
    if h / w > aspect_standard:
        target_w = int(h / aspect_standard)
        offset_w = (target_w - w) // 2
        rgb = np.pad(rgb, ((0, 0), (offset_w, offset_w), (0, 0)), mode="constant", constant_values=255)
        mask = np.pad(mask, ((0, 0), (offset_w, offset_w)), mode="constant", constant_values=0)
    else:
        target_h = int(w * aspect_standard)
        offset_h = (target_h - h) // 2
        rgb = np.pad(rgb, ((offset_h, offset_h), (0, 0), (0, 0)), mode="constant", constant_values=255)
        mask = np.pad(mask, ((offset_h, offset_h), (0, 0)), mode="constant", constant_values=0)

    rgb = rgb / 255.0
    mask_float = (mask / 255.0 > 0.5).astype(np.float32)
    rgb = rgb * mask_float[..., None] + (1 - mask_float[..., None])

    tgt_hw, _, _ = calc_new_tgt_size_by_aspect(rgb.shape[:2], aspect_standard, cfg.source_size, 14)
    rgb_resized = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((tgt_hw[1], tgt_hw[0]), Image.LANCZOS))
    rgb_tensor = torch.from_numpy(rgb_resized).float().permute(2, 0, 1) / 255.0
    return rgb_tensor.unsqueeze(0)

def create_c2w_matrix(position_np, rotation_q, eye='center'):
    position_corrected = np.array([position_np[0], -position_np[1], position_np[2]])
    rotation_corrected = np.array([-rotation_q[0], rotation_q[1], rotation_q[2], rotation_q[3]])
    rotation_matrix_unity = R.from_quat(rotation_corrected).as_matrix()

    c2w_unity = np.eye(4)
    c2w_unity[:3, :3] = rotation_matrix_unity
    c2w_unity[:3, 3] = position_corrected

    conversion_matrix = np.array([
        [1, 0,  0, 0],
        [0, 1,  0, 0],
        [0, 0, -1, 0],
        [0, 0,  0, 1],
    ])
    c2w_colmap = conversion_matrix @ c2w_unity

    offset = np.eye(4)
    if eye == 'left':
        offset[0, 3] = -CONFIG.IPD * 0.5
    elif eye == 'right':
        offset[0, 3] =  CONFIG.IPD * 0.5

    final_c2w = c2w_colmap @ offset
    final_w2c = np.linalg.inv(final_c2w)
    return torch.from_numpy(final_w2c).float()

def create_intr_matrix(width, height, fov_degrees):
    fov_rad = np.deg2rad(fov_degrees)
    fx = fy = (height / 2.0) / np.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    intr = torch.eye(4)
    intr[0, 0] = fx; intr[1, 1] = fy; intr[0, 2] = cx; intr[1, 2] = cy
    return intr

def smplx_tile_views(smplx_1x1: dict, B: int):
    out = {}
    for k, v in smplx_1x1.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        if k == 'betas':
            out[k] = v  # (1, nbeta) 그대로
        else:
            # (1,1,...) → (1,B,...) 로 타일
            if v.dim() >= 2 and v.shape[1] == 1:
                reps = [1, B] + [1] * (v.dim() - 2)
                out[k] = v.repeat(*reps).contiguous()
            else:
                out[k] = v
    return out

def _encode_jpeg_np(img_u8: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_u8, mode='RGB').save(buf, 'JPEG', quality=quality, optimize=False)
    return buf.getvalue()

def load_models_and_avatar():
    print("🚀 Starting model loading and avatar generation...")
    PartialState()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    original_dir = os.getcwd()
    os.chdir(CONFIG.LHM_APP_PATH)
    try:
        cfg = parse_configs_local()
        g["lhm"] = build_model_local(cfg).to(device)

        pose_estimator = PoseEstimator("./pretrained_models/human_model_files/", device=device)
        facedetector   = VGGHeadDetector("./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd", device=device)
        parsing_net    = SAM2Seg() if SAM2Seg else None

        target_image_rel_path = os.path.relpath(os.path.join(original_dir, CONFIG.TARGET_IMAGE_PATH), os.getcwd())
        with torch.no_grad():
            if parsing_net:
                parsing_out = parsing_net(img_path=target_image_rel_path, bbox=None)
                parsing_mask = (parsing_out.masks * 255).astype(np.uint8)
            else:
                rgba = np.array(Image.open(target_image_rel_path).convert("RGBA"))
                parsing_mask = rgba[..., 3]

            shape_pose = pose_estimator(target_image_rel_path)
            assert shape_pose.is_full_body, "Input image is not a full body shot."

            image = infer_preprocess_image_local(target_image_rel_path, parsing_mask)

            # src_head_rgb
            try:
                rgb = np.array(Image.open(target_image_rel_path))[..., :3]
                rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
                bbox = facedetector.detect_face(rgb_tensor)
                head_rgb = rgb_tensor[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                head_rgb = head_rgb.permute(1, 2, 0)
                src_head_rgb_np = head_rgb.cpu().numpy()
            except Exception as e:
                print(f"w/o head input! Using blank image: {e}")
                src_head_rgb_np = np.zeros((112, 112, 3), dtype=np.uint8)

            try:
                src_head_rgb_np = cv2.resize(src_head_rgb_np, dsize=(cfg.src_head_size, cfg.src_head_size),
                                             interpolation=cv2.INTER_AREA)
            except:
                src_head_rgb_np = np.zeros((cfg.src_head_size, cfg.src_head_size, 3), dtype=np.uint8)

            src_head_rgb = torch.from_numpy(src_head_rgb_np / 255.0).float().permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            shape_param = torch.tensor(shape_pose.beta, dtype=torch.float32).unsqueeze(0).to(device)
            smplx_params = {
                'betas': shape_param,
                'root_pose': torch.zeros(1, 1, 3, device=device),
                'body_pose': torch.zeros(1, 1, 21, 3, device=device),
                'jaw_pose':  torch.zeros(1, 1, 3, device=device),
                'leye_pose': torch.zeros(1, 1, 3, device=device),
                'reye_pose': torch.zeros(1, 1, 3, device=device),
                'lhand_pose': torch.zeros(1, 1, 15, 3, device=device),
                'rhand_pose': torch.zeros(1, 1, 15, 3, device=device),
                'trans': torch.zeros(1, 1, 3, device=device),
            }
            g["smplx_params"] = smplx_params

            dummy_c2ws = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
            dummy_intrs = create_intr_matrix(1, 1, 90).to(device).unsqueeze(0).unsqueeze(0)
            dummy_bg    = torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(0).unsqueeze(0)

            g["gs_model_list"], g["query_points"], g["transform_mat_neutral_pose"] = g["lhm"].infer_single_view(
                image.unsqueeze(0).to(device),
                src_head_rgb.unsqueeze(0).to(device),
                None, None,
                render_c2ws=dummy_c2ws, render_intrs=dummy_intrs,
                render_bg_colors=dummy_bg, smplx_params=g["smplx_params"]
            )

        g["bg_color"] = torch.tensor([0.0, 0.0, 0.0], device=device).float()
        # g["bg_color"] = torch.tensor([1.0, 1.0, 1.0], device=device).float()
        print("✅ Models loaded and avatar generated successfully!")
    finally:
        os.chdir(original_dir)

# -----------------------------
# FastAPI
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models_and_avatar()
    yield
    print("Server shutting down.")

app = FastAPI(lifespan=lifespan)

class CameraData(BaseModel):
    pos_x: float; pos_y: float; pos_z: float
    rot_x: float; rot_y: float; rot_z: float; rot_w: float
    eye: str

class SmplxData(BaseModel):
    root_pose: List[float]
    trans: List[float]
    body_pose: List[List[float]]
    lhand_pose: List[List[float]]
    rhand_pose: List[List[float]]

class VRAvatarData(BaseModel):
    camera: CameraData
    smplx: SmplxData

class VRAvatarBatch(BaseModel):
    cameras: List[CameraData]
    smplx: SmplxData

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 WebSocket connection established.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    FPS_LOG_SEC = 2.0
    win_start = perf_counter()
    eye_frames = 0
    bytes_sent = 0
    acc_render = 0.0
    acc_encode = 0.0
    acc_send   = 0.0

    try:
        while True:
            data = await websocket.receive_json()
            if isinstance(data, dict) and "cameras" in data:
                batch = VRAvatarBatch(**data)
                cams = batch.cameras
                B = len(cams)

                # (1,B,4,4) intrs
                fov_rad = np.deg2rad(CONFIG.FOV_DEGREES)
                fx = fy = (CONFIG.RENDER_HEIGHT / 2.0) / np.tan(fov_rad / 2.0)
                cx = CONFIG.RENDER_WIDTH / 2.0
                cy = CONFIG.RENDER_HEIGHT / 2.0
                intr_base = torch.eye(4, device=device)
                intr_base[0,0] = fx; intr_base[1,1] = fy; intr_base[0,2] = cx; intr_base[1,2] = cy
                render_intrs = intr_base.unsqueeze(0).unsqueeze(0).repeat(1, B, 1, 1)  # (1,B,4,4)

                # (1,B,4,4) c2w + headers
                header_list = []
                c2w_list = []
                for cam in cams:
                    pos_np = np.array([cam.pos_x, cam.pos_y, cam.pos_z], dtype=np.float32)
                    rot_q_np = np.array([cam.rot_x, cam.rot_y, cam.rot_z, cam.rot_w], dtype=np.float32)
                    w2c = create_c2w_matrix(pos_np, rot_q_np, cam.eye).to(device)
                    c2w_list.append(w2c)
                    header_list.append(b'\x00' if cam.eye.lower() == 'left' else b'\x01')
                render_c2ws = torch.stack(c2w_list, dim=0).unsqueeze(0)  # (1,B,4,4)

                # (1,B,3) bg
                bg_colors = g["bg_color"].reshape(1,1,3).to(device).repeat(1, B, 1)

                smplx_info = batch.smplx
                dynamic_smplx_params = {
                    'betas': g['smplx_params']['betas'],  # (1, nbeta)
                    'transform_mat_neutral_pose': g['transform_mat_neutral_pose'],

                    'root_pose':  torch.tensor([smplx_info.root_pose], device=device).float().unsqueeze(0),  # (1,1,3)
                    'body_pose':  torch.tensor([smplx_info.body_pose], device=device).float().unsqueeze(0),  # (1,1,21,3)
                    'lhand_pose': torch.tensor([smplx_info.lhand_pose], device=device).float().unsqueeze(0), # (1,1,15,3)
                    'rhand_pose': torch.tensor([smplx_info.rhand_pose], device=device).float().unsqueeze(0), # (1,1,15,3)
                    'trans':      torch.zeros(1, 1, 3, device=device),

                    'jaw_pose':   torch.zeros(1, 1, 3, device=device),
                    'leye_pose':  torch.zeros(1, 1, 3, device=device),
                    'reye_pose':  torch.zeros(1, 1, 3, device=device),
                    'focal':      torch.tensor([[[fx, fy]]], device=device, dtype=torch.float32),      # (1,1,2)
                    'princpt':    torch.tensor([[[cx, cy]]], device=device, dtype=torch.float32),      # (1,1,2)
                    'img_size_wh':torch.tensor([[[CONFIG.RENDER_WIDTH, CONFIG.RENDER_HEIGHT]]],
                                               device=device, dtype=torch.float32),                   # (1,1,2)
                    'expr':       torch.zeros(1, 1, 100, device=device)
                }
                smplx_params_B = smplx_tile_views(dynamic_smplx_params, B)  # (1,B,...)로

                imgs = []
                t0_all = perf_counter()
                with torch.no_grad():
                    for view_idx in range(B):
                        single_smpl = g["lhm"].renderer.get_single_view_smpl_data(smplx_params_B, view_idx)
                        res = g["lhm"].renderer.forward_animate_gs(
                            g["gs_model_list"], g["query_points"], single_smpl,
                            render_c2ws[:, view_idx:view_idx+1],   # (1,1,4,4)
                            render_intrs[:, view_idx:view_idx+1],  # (1,1,4,4)
                            CONFIG.RENDER_HEIGHT, CONFIG.RENDER_WIDTH,
                            bg_colors[:, view_idx:view_idx+1]      # (1,1,3)
                        )
                        comp = res["comp_rgb"]  # (1,1,3,H,W) (torch, 0~1)
                        rgb = comp[0,0].permute(1,2,0).contiguous().cpu().numpy()  # (H,W,3), float[0..1]
                        imgs.append(np.clip(rgb, 0, 1))
                t1_all = perf_counter()
                per_eye_render = (t1_all - t0_all) / max(B,1)

                arr_u8 = (np.stack(imgs, axis=0) * 255.0).astype(np.uint8)  # (B,H,W,3)
                t2 = perf_counter()
                jpg_list = await asyncio.gather(*[
                    asyncio.to_thread(_encode_jpeg_np, arr_u8[i].copy(), CONFIG.JPEG_QUALITY)
                    for i in range(B)
                ])
                t3 = perf_counter()

                for i in range(B):
                    payload = header_list[i] + jpg_list[i]
                    t_send0 = perf_counter()
                    await websocket.send_bytes(payload)
                    t_send1 = perf_counter()

                    eye_frames += 1
                    bytes_sent += len(payload)
                    acc_render += per_eye_render
                    acc_encode += (t3 - t2) / max(B,1)
                    acc_send   += (t_send1 - t_send0)

                now = perf_counter()
                if now - win_start >= FPS_LOG_SEC:
                    fps = eye_frames / (now - win_start)
                    avg_r = (acc_render / max(eye_frames, 1)) * 1000.0
                    avg_e = (acc_encode / max(eye_frames, 1)) * 1000.0
                    avg_s = (acc_send   / max(eye_frames, 1)) * 1000.0
                    mbps  = (bytes_sent * 8) / (now - win_start) / 1e6
                    print(f"[SRV] FPS {fps:5.1f} | render {avg_r:6.1f} ms | encode {avg_e:6.1f} ms | send {avg_s:6.1f} ms | net {mbps:5.1f} Mbps")
                    win_start = now
                    eye_frames = 0
                    bytes_sent = 0
                    acc_render = acc_encode = acc_send = 0.0

            else:
                avatar_data = VRAvatarData(**data)
                cam = avatar_data.camera

                pos_np = np.array([cam.pos_x, cam.pos_y, cam.pos_z], dtype=np.float32)
                rot_q  = np.array([cam.rot_x, cam.rot_y, cam.rot_z, cam.rot_w], dtype=np.float32)
                render_c2ws = create_c2w_matrix(pos_np, rot_q, cam.eye)  # (4,4)

                fov_rad = np.deg2rad(CONFIG.FOV_DEGREES)
                fx = fy = (CONFIG.RENDER_HEIGHT / 2.0) / np.tan(fov_rad / 2.0)
                cx = CONFIG.RENDER_WIDTH / 2.0
                cy = CONFIG.RENDER_HEIGHT / 2.0
                intr = torch.eye(4, device=device)
                intr[0,0] = fx; intr[1,1] = fy; intr[0,2] = cx; intr[1,2] = cy

                with torch.no_grad():
                    smplx_info = avatar_data.smplx
                    dynamic_smplx_params = {
                        'betas': g['smplx_params']['betas'],
                        'transform_mat_neutral_pose': g['transform_mat_neutral_pose'],
                        'root_pose':  torch.tensor([smplx_info.root_pose], device=device).float().unsqueeze(0),
                        'body_pose':  torch.tensor([smplx_info.body_pose], device=device).float().unsqueeze(0),
                        'lhand_pose': torch.tensor([smplx_info.lhand_pose], device=device).float().unsqueeze(0),
                        'rhand_pose': torch.tensor([smplx_info.rhand_pose], device=device).float().unsqueeze(0),
                        'trans':      torch.zeros(1, 1, 3, device=device),
                        'jaw_pose':   torch.zeros(1, 1, 3, device=device),
                        'leye_pose':  torch.zeros(1, 1, 3, device=device),
                        'reye_pose':  torch.zeros(1, 1, 3, device=device),
                        'focal':      torch.tensor([[[fx, fy]]], device=device, dtype=torch.float32),
                        'princpt':    torch.tensor([[[cx, cy]]], device=device, dtype=torch.float32),
                        'img_size_wh':torch.tensor([[[CONFIG.RENDER_WIDTH, CONFIG.RENDER_HEIGHT]]],
                                                   device=device, dtype=torch.float32),
                        'expr':       torch.zeros(1, 1, 100, device=device)
                    }

                    t0 = perf_counter()
                    single_smpl_data = g["lhm"].renderer.get_single_view_smpl_data(dynamic_smplx_params, 0)
                    res = g["lhm"].renderer.forward_animate_gs(
                        g["gs_model_list"], g["query_points"], single_smpl_data,
                        render_c2ws.unsqueeze(0).unsqueeze(0).to(device),
                        intr.unsqueeze(0).unsqueeze(0).to(device),
                        CONFIG.RENDER_HEIGHT, CONFIG.RENDER_WIDTH,
                        g["bg_color"].reshape(1,1,3).to(device)
                    )
                    t1 = perf_counter()

                comp_rgb = res["comp_rgb"][0,0].permute(1,2,0).contiguous().cpu().numpy()
                comp_rgb = (np.clip(comp_rgb, 0, 1) * 255).astype(np.uint8)

                t2 = perf_counter()
                buf = io.BytesIO()
                Image.fromarray(comp_rgb).save(buf, 'JPEG', quality=CONFIG.JPEG_QUALITY, optimize=False)
                payload = (b'\x00' if cam.eye.lower() == 'left' else b'\x01') + buf.getvalue()
                t3 = perf_counter()
                await websocket.send_bytes(payload)
                t4 = perf_counter()

                eye_frames += 1
                bytes_sent += len(payload)
                acc_render += (t1 - t0)
                acc_encode += (t3 - t2)
                acc_send   += (t4 - t3)

                now = perf_counter()
                if now - win_start >= FPS_LOG_SEC:
                    fps = eye_frames / (now - win_start)
                    avg_r = (acc_render / max(eye_frames, 1)) * 1000.0
                    avg_e = (acc_encode / max(eye_frames, 1)) * 1000.0
                    avg_s = (acc_send   / max(eye_frames, 1)) * 1000.0
                    mbps  = (bytes_sent * 8) / (now - win_start) / 1e6
                    print(f"[SRV] FPS {fps:5.1f} | render {avg_r:6.1f} ms | encode {avg_e:6.1f} ms | send {avg_s:6.1f} ms | net {mbps:5.1f} Mbps")
                    win_start = now
                    eye_frames = 0
                    bytes_sent = 0
                    acc_render = acc_encode = acc_send = 0.0

    except WebSocketDisconnect:
        print("🔌 WebSocket connection closed.")
    except Exception as e:
        print(f"💥 An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
