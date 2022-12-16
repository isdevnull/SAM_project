# SAM Project

This project explores Sharpness-Aware Minimization (SAM) method.

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

### Reference
SAM method is taken from the following repository:
https://github.com/davda54/sam

### Authors
- Ivan Shchekotov
- Kirill Tamogashev
- Emil Alkin