## Prerequisites

* Python 2.7
* Pytorch 0.4
* CUDA 9.0

## Installation the environment

Please first refer to [MattNet](https://github.com/insomnia94/MAttNet) to prepare related data. Replace all files in this repository.

## Pre-trained models
All pre-trained models and related data can be downloaded [here](https://drive.google.com/drive/folders/16j1X5ldVeAxgU71pvXtW-yv6B2ABI8y5?usp=sharing). 

## Generate the Triads
You can generate the triad file thourgh our previous work [DTWREG](https://github.com/insomnia94/DTWREG).

## Supervised Training

```bash
python ./tools/train_SL.py
```

## Reinforcement Training

```bash
python ./tools/train_AC.py
```
## Evaluation

```bash
python ./eval.py
```
