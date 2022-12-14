# ARL

This is the implementation
of [Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-training with Knowledge](https://arxiv.org/abs/2209.07118)
at ACMMM-2022.

## Table of Contents

- [Requirements](#requirements)
- [Pre-training](#pre-training)
- [Downstream Evaluation](#downstream-evaluation)

## Requirements

Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

Note: ARL involves the knowledge extraction and knowledge integration, which need more packages. Therefore, please be
patient to install the environment. :-)

## Pre-training

### 1. Dataset Preparation

Please organize the pre-training datasets as the following structure:

```angular2
root:[data]
+--pretrain_data
| +--roco
| | +--train
| | +--val
| | +--test
| +--medicat
| | +--net
| | +--release
| +--mimic_cxr
| | +--files
| | +--mimic_cxr_sectioned.csv
| | +--mimic-cxr-2.0.0-split.csv
| | +--mimic-cxr-2.0.0-metadata.csv
| | +--mimic-cxr-2.0.0-chexpert.csv
```

### 2. Pre-processing

Run the following command to pre-process the data:

```angular2
python prepro/prepro_pretraining_data.py
```

to get the following arrow files:

```angular2
root:[data]
+--pretrain_arrows
| +--medicat_train.arrow
| +--medicat_val.arrow
| +--medicat_test.arrow
| +--roco_train.arrow
| +--roco_val.arrow
| +--roco_test.arrow
| +--mimic_cxr_train.arrow
| +--mimic_cxr_val.arrow
| +--mimic_cxr_test.arrow
```

### 3. Download the initialized weights for pre-training

Download the initialized meter
weights [here](https://drive.google.com/drive/folders/1PiXnT65WR8qb6VAqE1lwidLrOLThuqY7?usp=share_link).

### 4. Pre-training

Now we can start to pre-train the arl model:

```angular2
bash run_scripts/pretrain_arl.sh
```

## Downstream Evaluation

### 1. Dataset Preparation

Please organize the fine-tuning datasets as the following structure:

```angular2
root:[data]
+--finetune_data
| +--melinda
| | +--train.csv
| | +--dev.csv
| | +--test.csv
| | +--melinda_images
| +--slack
| | +--train.json
| | +--validate.json
| | +--test.json
| | +--imgs
| +--vqa_rad
| | +--trainset.json
| | +--valset.json
| | +--testset.json
| | +--images
| +--medvqa_2019
| | +--val
| | +--test
| | +--train
```

### 2. Pre-processing

Run the following command to pre-process the data:

```angular2
python prepro/prepro_finetuning_data.py
```

to get the following arrow files:

```angular2
root:[data]
+--finetune_arrows
| +--vqa_vqa_rad_train.arrow
| +--vqa_vqa_rad_val.arrow
| +--vqa_vqa_rad_test.arrow
| +--vqa_slack_train.arrow
| +--vqa_slack_test.arrow
| +--vqa_slack_val.arrow
| +--vqa_medvqa_2019_train.arrow
| +--vqa_medvqa_2019_val.arrow
| +--vqa_medvqa_2019_test.arrow
| +--cls_melinda_train.arrow
| +--cls_melinda_val.arrow
| +--cls_melinda_test.arrow
| +--irtr_roco_train.arrow
| +--irtr_roco_val.arrow
| +--irtr_roco_test.arrow
```

### 3. Fine-Tuning

Now you can start to fine-tune the arl model:

```angular2
bash run_scripts/finetune_arl.sh
```

### 4. Test

You can start to test our fine-tuned models directly:

```angular2
bash run_scripts/test_arl.sh
```

NOET: This is a good way to check whether your environment is set up in the same way as ours (if you can reproduce the
same results).

## Acknowledgement

The code is based on [OpenKE](https://github.com/thunlp/OpenKE), [ViLT](https://github.com/dandelin/ViLT)
, [METER](https://github.com/zdou0830/METER)
and [MAE](https://github.com/facebookresearch/mae).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citations

If ARL is useful for your research, please consider citing:

```angular2
@inproceedings{chen2022arl,
  title={Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-training with Knowledge},
  author={Chen, Zhihong and Li, Guanbin and Wan, Xiang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```
