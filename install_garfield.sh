conda create -n garfield_12 python=3.10 -y
conda activate garfield_12

# conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 -y

conda install -c "nvidia/label/cuda-12.0.0" cuda-toolkit -y




pip install ninja 
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

cd ~/Research/nerfstudio
pip install -e .

# conda install -c rapidsai -c conda-forge -c nvidia cuml -y
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.2.* cuml-cu12==24.2.* -y

cd ~/Research/garfield
pip install -e .
pip install git+https://github.com/facebookresearch/segment-anything.git
