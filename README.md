# SRGAN-PyTorch

## Overview
### Problem Statement
Generation of Hazard map at 1m grid spacing (1m height resolution) using 5m spatial resolution data for safely navigating a Lander to a safe landing site using super resolution techniques.

We have TMC images of nearly 80% area of moon at 5m resolution while OHRC data at 25cm resolution having very limited coverage. Due to limited coverage of OHRC, there are constrains to land at any place on the moon surface. Hence, a problem has been defined to create hazard map using super resolution techniques from TMC 5m images considering the hazard definitions (like Slope-10 degree, Crater / Boulder depth/height – 1m, Crater distribution, shadow etc.) for safely navigating a Lander. This challenge includes the showcasing of lander navigation techniques for safe landing considering reference as TMC 5m datasets in near real time.

## Walkthrough Filesystem

- `model.py` : model architecture
- `train_srgan.py` : run this file to train from scratch or continue training of the SRGAN model
- `generate_srgan.py` : run this file to generate a 16 times superscaled image from TMC2 image
- `evaluation.py` : run this file to evaluate our 16 times upsampled data with test ohrc over SSIM evalaution metric
- `srgan_config.py` : configure training or generating parameters here
- `dataset.py` : creates batches of tensors from training or testing data using dataloader
- `image_quality_assessment.py` : code for evaluation metrics
- `requirements.txt` : environment requirements
- `cascade.py` : premature version of generate.py

- `train_data` : strore train data here
- `pretrained_weights` : strore pretrained weights here
- `results` : best and latest weights for both generator and discrimniator while training will be saved here
- `samples` : logs and per epoch weights can be seen here


## Pretrained Weights

- g_model weights : `g_last.pth.tar`
- d_model weights : `d_last.pth.tar`
- These can be used for the purpose of further training the model over custom dataset.
- g_model weights : `g_best.pth.tar`

## Datasets
Chandrayan TMC-2 : Apollo-12 and Apollo-16 from ISSDC website
NASA LRO: Apollo-12 and Apollo-16 from LROC-NAC website

## How to Train, Generate and Evaluate

All three training, testing and evaluation only need to modify the `./srgan_config.py` file.
For all the bash commands current working directory is expected to be `SRGAN-implementation`

### Generate SuperResolution images

Modify the `srgan_config.py` file.

- line 29: `mode` change to `generate`.
- line 31: `exp_name` change to a new experiment name each time training.
- line 102: `g_model_weights_path` change to `./pretrained_models/generate/g_best.pth.tar`.
- the input low resoltion files must be of dimension AxA where A is a multiple of 24
- store the input low resoltion files in the directory `./generate_data/TMC2/dim_1x`.
- the corresponding output images can be found in the directory `./generate_data/TMC2/dim_16x`.

```bash
python generate_srgan.py
```

### Train SRGAN model

Modify the `srgan_config.py` file.

- line 29: `mode` change to `train`.
- line 31: `exp_name` change to a new experiment name each time training.
- line 47: `pretrained_d_model_weights_path` change to `./pretrained_models/train/d_last.pth.tar`.
- line 48: `pretrained_g_model_weights_path` change to `./pretrained_models/train/g_last.pth.tar`.
- best and last trained weights of both generator and discrimnator would be saved in `./results/{exp_name}`.

```bash
python train_srgan.py
```

### Evaluation

Modify the `srgan_config.py` file.

- line 29: `mode` change to `evaluate`.

```bash
python evaluation.py
```

- this unzipped folder can also be used to run evaluation over other metrics
- per image size in this folder is 3840x3840 8-bit images


### Installations

```bash
pip install opencv-python numpy tqdm torch torchvision natsort typing scipy
## or
pip install -r requirements.txt
```
