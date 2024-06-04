<div align="center">

# CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition

[![arXiv](https://img.shields.io/badge/arXiv-2207.04410-b31b1b.svg)](https://arxiv.org/abs/2207.04410)

</div>

## Project structure

```bash
├── README.md
├── comer               # model definition folder
├── convert2symLG       # official tool to convert latex to symLG format
├── lgeval              # official tool to compare symLGs in two folder
├── config.yaml         # config for CoMER hyperparameter
├── data.zip
├── eval_all.sh         # script to evaluate model on all CROHME test sets
├── example
│   ├── UN19_1041_em_595.bmp
│   └── example.ipynb   # HMER demo
├── lightning_logs      # training logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=151-step=57151-val_ExpRate=0.6365.ckpt
│       ├── config.yaml
│       └── hparams.yaml
├── requirements.txt
├── scripts             # evaluation scripts
├── setup.cfg
├── setup.py
└── train.py
```

## Install dependencies

```bash
cd CoMER
# install project   
conda create -y -n CoMER python=3.10
conda activate CoMER
# ubuntu==20.04, cuda==11.8,
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
```

## Training

If you want to reproduce our optimizations, please follow these steps to make the changes:

* Random Rotation:
  In  `comer\datamodule\dataset.py`, `AUG_MAX_ANGLE`(line 14) is the maximum angle of rotation. And you need to make sure that line 24 is not commented.
* Horizontal Stretching:
  In  `comer\datamodule\dataset.py`, you need to make sure that line 25 is not commented.
* Parallel DenseNets:
  In `comer\model\encoder.py`, you need to make sure that the first `class Encoder` is commented and the second `class Encoder` is not.

Next, navigate to CoMER folder and run `train.py`. It may take **7~8** hours on **4** NVIDIA 2080Ti gpus using ddp.

```bash
# train CoMER(Fusion) model using 4 gpus and ddp
python train.py --config config.yaml  
```

You may change the `config.yaml` file to train different models

```yaml
# train BTTR(baseline) model
cross_coverage: false
self_coverage: false

# train CoMER(Self) model
cross_coverage: false
self_coverage: true

# train CoMER(Cross) model
cross_coverage: true
self_coverage: false

# train CoMER(Fusion) model
cross_coverage: true
self_coverage: true
```

For single gpu user, you may change the `config.yaml` file to

```yaml
gpus: 1
# gpus: 4
# accelerator: ddp
```

## Evaluation

Metrics used in validation during the training process is not accurate.

For accurate metrics reported in the paper, please use tools officially provided by CROHME 2019 oganizer:

A trained CoMER(Fusion) weight checkpoint has been saved in `lightning_logs/version_0`

```bash
perl --version  # make sure you have installed perl 5

unzip -q data.zip

sudo apt-get install libxml-libxml-perl

# evaluation
# evaluate model in lightning_logs/version_0 on all CROHME test sets
# results will be printed in the screen and saved to lightning_logs/version_0 folder
bash eval_all.sh 0
# If you want to evaluate the optimized model, please change the code above into the code below
bash eval_all.sh x	# x = 11 if you want to evaluate the model of random rotation of 10°
			# x = 12 if you want to evaluate the model of horizontal stretching 0.7 to 1.4 times
			# x = 14 if you want to evaluate the model of random rotation of 5°
			# x = 19 if you want to evaluate the model of random rotation of 5° and horizontal stretching 0.7 to 1.4 times
			# x = 20 if you want to evaluate the origin model after 50 epochs of training
			# x = 40 if you want to evaluate the model of parallel DenseNets
```
