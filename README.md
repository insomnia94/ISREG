## Prerequisites

* Python 2.7
* Pytorch 0.4
* CUDA 9.0

## Installation the environment

Please first refer to [MattNet](https://github.com/insomnia94/MAttNet) and [KPRN](https://github.com/GingL/KPRN) to prepare related data. Then replace all files in this repository. "loaders" and "evals" should be placed in the "./lib/" and the main model file "model.py" should be placed in "./lib/layers/".

## Pre-trained models
All pre-trained models and related data can be downloaded [here](https://drive.google.com/drive/folders/16j1X5ldVeAxgU71pvXtW-yv6B2ABI8y5?usp=sharing). "mrcn_cmr_with_st.json","mrcn_cmr_with_st.pth","mrcn_cmr_with_st_critic.pth","mrcn_cmr_with_st_Re.pth" should be replaced in "./output/dataset_split/id_num/". "sent_extract.json" should be replaced in "./cache/sub_obj_wds/dataset_split/".

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
