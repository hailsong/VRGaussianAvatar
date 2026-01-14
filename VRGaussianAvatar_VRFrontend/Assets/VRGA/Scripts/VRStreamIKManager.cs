// VRStreamIKManager.cs
using UnityEngine;
using System.Collections.Concurrent;
using System;
using System.Collections.Generic;
using System.IO;                      // For dual-pack binary parsing
using RootMotion.FinalIK;
using Newtonsoft.Json;
using NativeWebSocket;

public class VRStreamIKManager : MonoBehaviour
{
    [Header("Server Settings")]
    public string serverUrl = "ws://127.0.0.1:8000/ws";

    [Header("Target Camera Rig")]
    public Transform cameraRigTransform;
    public Transform offsetTransform;

    [Header("Avatar Settings")]
    public VRIK ik;
    public Animator animator;

    [Header("VR Display Settings")]
    public Material leftEyeMaterial;
    public Material rightEyeMaterial;
    public float positionScaleFactor = 1.0f;

    [Header("Performance/Protocol")]
    [Tooltip("true: server_dualpack.py (dual-pack) / false: legacy server (per-eye frames)")]
    public bool useBatch = true;
    [Tooltip("Drop dual-pack bundle if frame order is reversed")]
    public bool dropOutOfOrder = true;

    // Batch state
    private bool isBatchRequesting = false;
    private int batchExpected = 0, batchReceived = 0;

    [Header("Debugging")]
    [Tooltip("Log root/spine poses every 1 second")]
    public bool enablePoseDebug = true;

    // ---------------- Serialization Structures ----------------
    [Serializable]
    public class SmplxData
    {
        public List<float> root_pose = new List<float>(3);
        public List<float> trans = new List<float>(3);
        public List<List<float>> body_pose = new List<List<float>>();
        public List<List<float>> lhand_pose = new List<List<float>>();
        public List<List<float>> rhand_pose = new List<List<float>>();
    }

    [Serializable]
    public class CameraData
    {
        public float pos_x, pos_y, pos_z;
        public float rot_x, rot_y, rot_z, rot_w;
        public string eye; // "left" / "right"
    }

    [Serializable]
    public class VRAvatarData   // Legacy (per-eye)
    {
        public CameraData camera;
        public SmplxData smplx;
    }

    [Serializable]
    public class VRAvatarBatch  // Dual-pack (batch)
    {
        public int frame_id;            // ¡Ú required by server_dualpack
        public List<CameraData> cameras;
        public SmplxData smplx;
    }

    // ---------------- Private ----------------
    private WebSocket websocket;
    private Texture2D leftEyeTexture;
    private Texture2D rightEyeTexture;

    // messageQueue: (eyeId, jpegBytes)
    private readonly ConcurrentQueue<Tuple<byte, byte[]>> messageQueue = new ConcurrentQueue<Tuple<byte, byte[]>>();

    private bool isLeftRequesting = false;
    private bool isRightRequesting = false;

    private float debugTimer = 0f;
    private float debugLogInterval = 1.0f;

    // Store initial T-pose local rotations
    private Dictionary<HumanBodyBones, Quaternion> initialLocalRotations;

    // SMPL-X joint mapping
    private readonly HumanBodyBones[] smplxBodyJoints = new HumanBodyBones[] {
        HumanBodyBones.Hips, HumanBodyBones.LeftUpperLeg, HumanBodyBones.RightUpperLeg,
        HumanBodyBones.Spine, HumanBodyBones.LeftLowerLeg, HumanBodyBones.RightLowerLeg,
        HumanBodyBones.Chest, HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot,
        HumanBodyBones.UpperChest, HumanBodyBones.LeftToes, HumanBodyBones.RightToes,
        HumanBodyBones.Neck, HumanBodyBones.LeftShoulder, HumanBodyBones.RightShoulder,
        HumanBodyBones.Head, HumanBodyBones.LeftUpperArm, HumanBodyBones.RightUpperArm,
        HumanBodyBones.LeftLowerArm, HumanBodyBones.RightLowerArm, HumanBodyBones.LeftHand,
        HumanBodyBones.RightHand,
    };

    // Dual-pack frame id
    private int _frameId = 0;
    private int _lastFrameId = -1;   // for dropping out-of-order frames

    // ---------------- Unity lifecycle ----------------
    async void Start()
    {
        leftEyeTexture = new Texture2D(2, 2);
        rightEyeTexture = new Texture2D(2, 2);

        if (cameraRigTransform == null || ik == null || animator == null)
        {
            Debug.LogError("Required components are not assigned! Disabling script.");
            enabled = false;
            return;
        }

        // Cache initial T-pose local rotations
        ik.solver.IKPositionWeight = 0f;
        CacheInitialRotations();
        ik.solver.IKPositionWeight = 1f;

        websocket = new WebSocket(serverUrl);
        websocket.OnOpen += () => Debug.Log("WebSocket Connection open!");
        websocket.OnError += (e) => Debug.LogError("WebSocket Error: " + e);
        websocket.OnClose += (e) => Debug.Log("WebSocket Connection closed!");

        // Server response: prioritize dual-pack ("VR"), otherwise legacy (eyeId + JPEG)
        websocket.OnMessage += (bytes) =>
        {
            if (bytes == null || bytes.Length == 0) return;

            // Dual-pack signature: 'V'(0x56) 'R'(0x52), header >= 9 bytes
            if (bytes.Length >= 9 && bytes[0] == 0x56 && bytes[1] == 0x52)
            {
                try
                {
                    using (var ms = new MemoryStream(bytes))
                    using (var br = new BinaryReader(ms))
                    {
                        var magic = br.ReadBytes(2);         // "VR"
                        ushort version = br.ReadUInt16();    // <H
                        uint frameId = br.ReadUInt32();      // <I
                        byte count = br.ReadByte();          // <B

                        // Drop out-of-order frames (optional)
                        if (dropOutOfOrder && (int)frameId <= _lastFrameId)
                        {
                            return; // reversed bundle, skip
                        }
                        _lastFrameId = (int)frameId;

                        for (int i = 0; i < count; i++)
                        {
                            byte eyeId = br.ReadByte();      // <B (0=left,1=right)
                            uint length = br.ReadUInt32();   // <I
                            byte[] payload = br.ReadBytes((int)length);
                            messageQueue.Enqueue(new Tuple<byte, byte[]>(eyeId, payload));
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"Dual-pack parse failed: {ex.Message}");
                }
            }
            else
            {
                // Legacy: [eyeId (1 byte)] + JPEG
                if (bytes.Length > 1)
                {
                    byte eyeIdentifier = bytes[0];
                    byte[] imageData = new byte[bytes.Length - 1];
                    Buffer.BlockCopy(bytes, 1, imageData, 0, imageData.Length);
                    messageQueue.Enqueue(new Tuple<byte, byte[]>(eyeIdentifier, imageData));
                }
            }
        };

        Debug.Log($"Attempting to connect to server ({serverUrl})...");
        await websocket.Connect();
    }

    void LateUpdate()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
#endif
        ProcessReceivedTextures();   // Empty queue each frame to minimize delay

        if (websocket != null && websocket.State == WebSocketState.Open && cameraRigTransform != null)
        {
            SmplxData poseData = GetSmplxData();
            if (enablePoseDebug) Debug_CheckPoseUpdate(poseData);

            if (useBatch)
            {
                if (!isBatchRequesting)
                {
                    batchExpected = (leftEyeMaterial != null ? 1 : 0) + (rightEyeMaterial != null ? 1 : 0);
                    if (batchExpected > 0)
                    {
                        SendBatchRequest(poseData);  // Dual-pack JSON (with frame_id)
                        isBatchRequesting = true;
                        batchReceived = 0;
                    }
                }
            }
            else
            {
                // Legacy: per-eye requests
                if (leftEyeMaterial != null && !isLeftRequesting) SendFrameRequest("left", poseData);
                if (rightEyeMaterial != null && !isRightRequesting) SendFrameRequest("right", poseData);
            }
        }
    }

    private void OnDestroy()
    {
        if (websocket != null && websocket.State == WebSocketState.Open)
            websocket.Close();
    }

    // ---------------- Processing received data ----------------
    void ProcessReceivedTextures()
    {
        bool completedThisFrame = false;  // whether both eyes completed this loop

        while (messageQueue.TryDequeue(out var message))
        {
            byte eyeIdentifier = message.Item1;
            byte[] imageData = message.Item2;

            if (eyeIdentifier == 0) // left
            {
                leftEyeTexture.LoadImage(imageData);
                if (leftEyeMaterial != null) leftEyeMaterial.mainTexture = leftEyeTexture;
                if (!useBatch) isLeftRequesting = false;
            }
            else if (eyeIdentifier == 1) // right
            {
                rightEyeTexture.LoadImage(imageData);
                if (rightEyeMaterial != null) rightEyeMaterial.mainTexture = rightEyeTexture;
                if (!useBatch) isRightRequesting = false;
            }

            if (useBatch)
            {
                batchReceived++;
                if (batchReceived >= batchExpected)
                {
                    // Trigger next batch immediately
                    completedThisFrame = true;
                }
            }
        }

        // Both eyes (or expected count) received: send next batch immediately
        if (useBatch && completedThisFrame)
        {
            if (websocket != null && websocket.State == WebSocketState.Open)
            {
                var poseData = GetSmplxData();
                SendBatchRequest(poseData);
                isBatchRequesting = true;
                batchReceived = 0;
            }
            else
            {
                isBatchRequesting = false;
            }
        }
    }

    // ---------------- Sending: batch (dual-pack server) ----------------
    async void SendBatchRequest(SmplxData poseData)
    {
        if (websocket.State != WebSocketState.Open) return;

        Vector3 finalPosition = offsetTransform != null
            ? cameraRigTransform.position + cameraRigTransform.rotation * offsetTransform.position
            : cameraRigTransform.position;
        Quaternion finalRotation = offsetTransform != null
            ? cameraRigTransform.rotation * offsetTransform.rotation
            : cameraRigTransform.rotation;

        var cams = new List<CameraData>();
        if (leftEyeMaterial != null) cams.Add(new CameraData
        {
            pos_x = finalPosition.x * positionScaleFactor,
            pos_y = finalPosition.y * positionScaleFactor,
            pos_z = finalPosition.z * positionScaleFactor,
            rot_x = finalRotation.x,
            rot_y = finalRotation.y,
            rot_z = finalRotation.z,
            rot_w = finalRotation.w,
            eye = "left"
        });
        if (rightEyeMaterial != null) cams.Add(new CameraData
        {
            pos_x = finalPosition.x * positionScaleFactor,
            pos_y = finalPosition.y * positionScaleFactor,
            pos_z = finalPosition.z * positionScaleFactor,
            rot_x = finalRotation.x,
            rot_y = finalRotation.y,
            rot_z = finalRotation.z,
            rot_w = finalRotation.w,
            eye = "right"
        });

        var payload = new VRAvatarBatch
        {
            frame_id = _frameId++,   // required by server_dualpack
            cameras = cams,
            smplx = poseData
        };

        string json = JsonConvert.SerializeObject(payload);
        await websocket.SendText(json);
    }

    // ---------------- Sending: legacy (old server) ----------------
    async void SendFrameRequest(string eye, SmplxData poseData)
    {
        if (websocket.State != WebSocketState.Open) return;
        if (eye == "left") isLeftRequesting = true; else isRightRequesting = true;

        Vector3 finalPosition = offsetTransform != null
            ? cameraRigTransform.position + cameraRigTransform.rotation * offsetTransform.position
            : cameraRigTransform.position;
        Quaternion finalRotation = offsetTransform != null
            ? cameraRigTransform.rotation * offsetTransform.rotation
            : cameraRigTransform.rotation;

        VRAvatarData data = new VRAvatarData
        {
            camera = new CameraData
            {
                pos_x = finalPosition.x * positionScaleFactor,
                pos_y = finalPosition.y * positionScaleFactor,
                pos_z = finalPosition.z * positionScaleFactor,
                rot_x = finalRotation.x,
                rot_y = finalRotation.y,
                rot_z = finalRotation.z,
                rot_w = finalRotation.w,
                eye = eye
            },
            smplx = poseData
        };

        string jsonData = JsonConvert.SerializeObject(data);
        await websocket.SendText(jsonData);
    }

    // ---------------- Generate SMPL-X data ----------------
    SmplxData GetSmplxData()
    {
        var smplx = new SmplxData();
        Transform root = ik.references.pelvis;

        smplx.trans = ConvertVector3(root.position);
        smplx.root_pose = ConvertRotation(root.rotation);

        // body_pose: delta vs. initial T-pose, Axis-Angle (mapping x, -y, -z)
        for (int i = 1; i < smplxBodyJoints.Length; i++)
        {
            HumanBodyBones bone = smplxBodyJoints[i];
            Transform jointTransform = animator.GetBoneTransform(bone);
            if (jointTransform != null)
                smplx.body_pose.Add(ConvertLocalRotation(bone, jointTransform.localRotation));
            else
                smplx.body_pose.Add(new List<float> { 0f, 0f, 0f });
        }

        // Fingers: currently set to 0
        for (int i = 0; i < 15; i++)
        {
            smplx.lhand_pose.Add(new List<float> { 0f, 0f, 0f });
            smplx.rhand_pose.Add(new List<float> { 0f, 0f, 0f });
        }

        return smplx;
    }

    // ---------------- Utils/Debug ----------------
    void CacheInitialRotations()
    {
        initialLocalRotations = new Dictionary<HumanBodyBones, Quaternion>();
        foreach (HumanBodyBones bone in smplxBodyJoints)
        {
            Transform jt = animator.GetBoneTransform(bone);
            if (jt != null) initialLocalRotations[bone] = jt.localRotation;
        }
        Debug.Log("Stored initial T-pose joint rotations of avatar.");
    }

    // body_pose conversion: delta vs. T-pose ¡æ Axis-Angle, mapping (x, -y, -z)
    List<float> ConvertLocalRotation(HumanBodyBones bone, Quaternion currentLocalRotation)
    {
        Quaternion initialRot = initialLocalRotations.ContainsKey(bone) ? initialLocalRotations[bone] : Quaternion.identity;
        Quaternion deltaRotation = currentLocalRotation * Quaternion.Inverse(initialRot);

        deltaRotation.ToAngleAxis(out float angle, out Vector3 axis);
        if (float.IsNaN(axis.x)) return new List<float> { 0f, 0f, 0f };

        float angleRad = angle * Mathf.Deg2Rad;
        Vector3 aa = axis * angleRad;

        return new List<float> { aa.x, -aa.y, -aa.z };
    }

    // root_pose conversion: world rotation ¡æ corrected (x, -y, -z, w) ¡æ Axis-Angle
    List<float> ConvertRotation(Quaternion worldRotation)
    {
        Quaternion corrected = new Quaternion(worldRotation.x, -worldRotation.y, -worldRotation.z, worldRotation.w);
        return QuaternionToAxisAngle(corrected);
    }

    List<float> QuaternionToAxisAngle(Quaternion q)
    {
        q.ToAngleAxis(out float angle, out Vector3 axis);
        if (float.IsNaN(axis.x)) return new List<float> { 0f, 0f, 0f };
        Vector3 axisAngle = axis * (angle * Mathf.Deg2Rad);
        return new List<float> { axisAngle.x, axisAngle.y, axisAngle.z };
    }

    List<float> ConvertVector3(Vector3 v)
    {
        // Flip Z sign to match server create_c2w_matrix
        return new List<float> { v.x, v.y, -v.z };
    }

    void Debug_CheckPoseUpdate(SmplxData currentPose)
    {
        debugTimer += Time.deltaTime;
        if (debugTimer >= debugLogInterval)
        {
            var rootPoseVec = new Vector3(currentPose.root_pose[0], currentPose.root_pose[1], currentPose.root_pose[2]);
            var spinePoseVec = new Vector3(currentPose.body_pose[2][0], currentPose.body_pose[2][1], currentPose.body_pose[2][2]);

            Debug.Log($"<color=cyan>--- Pose Update ({debugLogInterval:F1}s) ---</color>\n" +
                      $"<b>Pelvis (root_pose):</b> {rootPoseVec.ToString("F4")}\n" +
                      $"<b>Spine  (body_pose[2]):</b> {spinePoseVec.ToString("F4")}\n" +
                      $"-----------------------------------------");
            debugTimer = 0f;
        }
    }
}
