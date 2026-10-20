# [SoICT 2026] Benchmarking Data-Free Continual Model Merging for Whole-Slide Image Survival Analysis

<!-- **Paper:** Link to be added. -->

## Overview

This repository provides an end-to-end benchmark of data-free continual model merging for whole-slide image survival analysis. It covers task-specific fine-tuning, sequential model merging, and evaluation of six merging strategies across four TCGA cancer cohorts under TASK-IL and CLASS-IL settings.

## Environment

```bash
conda create -n bcmm python=3.10 -y
conda activate bcmm
pip install -r requirements.txt
```

**Pretrained model:** We use the pretrained [TITAN model from MahmoodLab](https://huggingface.co/MahmoodLab/TITAN).

## Running Experiments

### Training

```bash
python train.py --config configs/forward.yaml
```

### Model Merging

```bash
python merge.py --config configs/forward.yaml --method opcm
```

Other merging methods can be selected with `--method`: `weight_average`, `task_arithmetic`, `ties`, `dare`, `dop`, `nufilt`, and `opcm_original`.

### Evaluation

**TASK-IL:**

```bash
python evaluate.py --config configs/forward.yaml --method opcm --mode taskIL
```
**CLASS-IL:**

```bash
python evaluate.py --config configs/forward.yaml --method opcm --mode classIL
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{tran2026bcmm,
  title     = {Benchmarking Data-Free Continual Model Merging for Whole-Slide Image Survival Analysis},
  author    = {Tran, Vu Minh and Bui, Doanh C. and Nguyen, Ma{\"i} K. and Nguyen, Khang},
  booktitle = {Proceedings of the 15th International Symposium on Information and Communication Technology (SOICT 2026)},
  publisher = {Springer},
  series    = {Communications in Computer and Information Science},
  year      = {2026},
}
```
