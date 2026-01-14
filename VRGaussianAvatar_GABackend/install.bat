

@echo off
echo Installing Torch 2.3.0...
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

echo Installing dependencies...
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17

echo Uninstalling basicsr to avoid conflicts...
pip uninstall -y basicsr
pip install git+https://github.com/XPixelGroup/BasicSR

echo Installing pytorch3d...
@REM pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.3.1cu121

echo Installing sam2...
pip install git+https://github.com/hitsz-zuoqi/sam2/

@REM echo Installing diff-gaussian-rasterization...
@REM pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/




