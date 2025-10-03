## Dependencies

### Core Dependencies
- **Python**: 3.10.x
- **CUDA**: 11.7 (compatible with CUDA 11.8+)
- **cuDNN**: 8.5.0+

### Main Packages
- **PyTorch 2.0.1+cu117**
  ```bash
  pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
  ```
- **Open3D 0.18.0**
  ```bash
  pip install open3d==0.18.0
  ```
- **TensorFlow 2.10.0** (for TensorBoard and related tools)
  ```bash
  pip install tensorflow==2.10.0 tensorboard==2.10.1
  ```
- **Flash Attention 2.3.5**
  ```bash
  pip install flash-attn==2.3.5 --no-build-isolation
  ```

### Required System Packages
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libcap-dev \
    libgl1-mesa-glx \
    libopenmpi-dev \
    libx11-6 \
    python3-dev \
    zlib1g-dev

# For Open3D visualization
sudo apt install -y \
    libgl1 \
    libegl1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6
```

### Python Dependencies
Install all Python dependencies:
```bash
pip install -r requirements.txt
```

### Optional Dependencies
- **SPlisHSPlasH 2.4.0** (for generating training data and fluid particle sampling)
  - Source: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH
  - Compile in *Release* mode

- **OpenVDB** (for creating surface meshes)
  ```bash
  # Install system dependencies
  sudo apt install -y \
      libopenvdb-dev \
      libtbb-dev \
      libboost-iostreams-dev \
      libboost-system-dev \
      libboost-filesystem-dev
      
  # Python bindings
  pip install openvdb
  ```

### Environment Notes
- The environment has been tested on **Ubuntu 20.04/22.04 LTS**
- For **GPU acceleration**, ensure you have compatible NVIDIA drivers installed
- For **Docker** support, use `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04` as base image
- For **WSL2** users, ensure proper GPU passthrough is configured

# Network Training

## Water Dataset Acquisition
To bypass the data generation process, precomputed training and validation datasets are available for download:

| Dataset       | Size | URL                                                          |
|---------------|------|--------------------------------------------------------------|
| Default Data  | 34GB | [Download](https://drive.google.com/file/d/1b3OjeXnsvwUAeUq2Z0lcrX7j9U7zLO07) |

The default training dataset was generated using scripts within this repository, while the validation dataset corresponds to the benchmark data employed in our study.

## Fuel Surface Dataset Generation
Data generation scripts reside in the `datasets` subdirectory. To generate training and validation datasets:

1. Configure the path to the SPlisHSPlasH `DynamicBoundarySimulator` in `datasets/splishsplash_config.py`
2. Execute the processing scripts from the `datasets` directory:  
   `Raw directories → .zst → .npz (→ .obj)`

### Orientation Variants
- **Simulation Box Rotation** (fixed gravity direction):  
  ```bash
  sh datasets/create_fuel_yemian_rotatebox.sh
  ```
- **Gravity Direction Rotation** (fixed box orientation):  
  ```bash
  sh datasets/create_fuel_yemian_rotategravity.sh
  ```

## Direct Execution with Precomputed ZST Files
```bash 
scripts/run_network_fueltank.py --weights=....pt \
                --scene=scripts/example_scene.json \
                --zst_dir=[zst_directory] \
                --output=[output_directory] \
                --num_steps=400
```

## Model Training

### Execution
Train the model using generated data by executing the appropriate script from the `scripts` directory:
```bash
# PyTorch implementation
scripts/train_network_torch.py default.yaml
```
Execution generates a training directory (e.g., `train_network_torch_default`) containing logs viewable via TensorBoard.
If you want to train on fuel datasets, it is recommended to train on the checkpoint pretrained on water datasets: 
Copy the ckpts/ckpt-52000.pt into the folder ```train_network_torch_default/checkpoints```, then the training will start by the step 52000.

### Configuration
The `default.yaml` file specifies training parameters including:
- Dataset paths
- Rotational augmentation settings
- Hyperparameters
- Optimization strategies

## Evaluating the network
To evaluate the network run the ```scripts/evaluate_network.py``` script like this
```bash
python scripts/evaluate_network.py --trainscript=scripts/train_network_torch.py \
        --cfg=scripts/default_TankI.yaml \
        --weights=.../ckpt.pt \
        --device cuda
```
```--trainscript``` is the corresponding training script,  ```--cfg``` is the evaluating configuration, and ```--weights``` is the path to the weight (checkpoint) that is to be evaluated.

This will create the file ```ckpt.pt_eval.json```, which contains the 
individual errors between frame pairs.

The script will also print the overall errors. The output should look like 
this if you use the generated the data:
```{'err_n1': 0.0010124884783484353, 'err_n2': 0.002481036043415467, 'emd_n1': 0.00013176361526100565, 'emd_n2': 0.00023006679901601526, 'whole_seq_err': 0.024410381430610074}```

## Running the pretrained model

The pretrained network weights are in ```.../ckpt.pt``` for PyTorch.
The following code runs the network on the example scene
```bash 
# run the pretrained model for single fluid
scripts/run_network.py ---weights=.../ckpt.pt \
                --scene=scenes/example_scene.json \
                --output=scenes/output/test \
                --num_steps=400 \
                train_network_torc.py
# run the pretrained model for multi fluids    
scripts/run_network_multiflulid.py --weights=.../ckpt.pt \
                --scene=scenes/example_scene.json \
                --output=scenes/output.test \
                --num_steps=400 \ 
                train_network_torch.py 
```

The script writes point clouds with the particle positions as .ply files, which can be visualized with Open3D.
Note that SPlisHSPlasH is required for sampling the initial fluid volumes from ```.obj``` files.

## Rendering

See the [scenes](scenes/README.md) directory for instructions on how to create and render the example scenes like the canyon.
