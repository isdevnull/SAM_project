# SAM Project

This project attempts to explore and compare Sharpness Aware Minimization (SAM) methods.
Our main goal is to compare generalization ability of SAM and Fast Gradient Sign Method (FGSM) on CIFAR10
and CIFAR100 datasets.
## Quick Start
### Install
```angular2html
pip install -r requirements.txt
```
### Training
Available setups are:
```angular2html
configs/
├── cifar10
│   ├── c10_resnet50_fgsm.yaml
│   ├── c10_resnet50_no_sam.yaml
│   └── c10_resnet50_sam.yaml
└── cifar100
    ├── c100_resnet50_fgsm.yaml
    ├── c100_resnet50_no_sam.yaml
    └── c100_resnet50_sam.yaml
```
We support training in 3 regimes: 
- Default (no_sam), which corresponds to default model training
- SAM, which employs SAM procedure
- FGSM, which employs FGSM for calculating optimal epsilon

Minimal example of launching training procedure:
```angular2html
python main.py config=configs/cifar10/c10_resnet50_sam.yaml general.device="cuda" general.exp_name="experiment_name"
```
You can check config structure in more detail in any yaml file. 
Any config is modifiable both from cli (upon launch) and yaml file.
`general.exp_name` is the name of the run in default `runs` directory for tensorboard logging.
It is also modifiable via config.

Upon first run you can provide argument `data.download=True` to download required dataset.

### Inference
You can inference results (test loss, top-1 accuracy, top-5 accuracy) via the following command:
```angular2html
python main.py config=configs/inference_config.yaml inference.ckpt="path_to_ckpt.pth" general.test_batch_size=1 data.dataset=$DATASET
```
- `$DATASET` is either "CIFAR10" or "CIFAR100"
- `inference.ckpt` is mandatory cli argument

### Checkpoints and training data
Checkpoints and training data are provided via Google Drive link: https://drive.google.com/drive/folders/1l5ipbxxJvmgadcfHDQ9oAeCXw1VE5sLt
You can download all data (careful, large size) via:
```angular2html
pip install gdown 
gdown --folder https://drive.google.com/drive/folders/1l5ipbxxJvmgadcfHDQ9oAeCXw1VE5sLt
```
Each folder contains run with corresponding tensorboard log data.
You can visualize results via:
```angular2html
tensorboard --logdir sam_project_data
```
Every run folder contains `checkpoints` folder containing model checkpoints that can be used for inference and verifying results.

## Reference
SAM method is taken from the following repository:
https://github.com/davda54/sam

## Authors
- Ivan Shchekotov
- Kirill Tamogashev
- Emil Alkin