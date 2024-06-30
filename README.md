# MEMO: Test Time Robustness via Adaptation and Augmentation 

Just using Random Resized Crop in augmentation ~ >10% accuracy in adversarial images!

## Overview

This project implements [**Marginal Entropy Minimization with One Test Point (MEMO)**](https://proceedings.neurips.cc/paper_files/paper/2022/file/fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf) for domain adaptation in image classification. It is based on the paper **"MEMO: Test Time Robustness via Adaptation and Augmentation"**, using the ImageNet-A dataset. The approach addresses domain shifts, adapting to each test sample individually without needing additional training data.

## Setup and Execution

### Requirements

To set up the project on Google Colab, use the following commands:

```bash
!pip install ftfy regex tqdm -q
!pip install git+https://github.com/openai/CLIP.git -q
!pip install wandb
```

## Running the Project

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
Ensure the root directory is set to the ImageNet-A folder under Notebook Configuration.
```bash
ImageNet-A/
│
├── n01440764/
│   ├── image1.JPEG
│   ├── image2.JPEG
│   └── ...
│
├── n01443537/
│   ├── image1.JPEG
│   ├── image2.JPEG
│   └── ...
│
├── n01514668/
│   ├── image1.JPEG
│   ├── image2.JPEG
│   └── ...
│
└── ...
```
For AWS SageMaker, refer to the memo_sagemaker notebook for specific instructions.

#### Enabling Weights & Biases

If you would like to enable wandb to track your experiments, 
set `wandb_active = True` and login to your account. It's free to use with university account. 

[Also you can see our augmentations here](https://wandb.ai/project-zero/imagenet-adaptation-zehra/reports/Augmentations-for-MEMO--Vmlldzo4NTA1NTA1?accessToken=u37q32nxru6y6vir0glo3e9h616qscb09hj40gx2tq25mv6c6rxckrxjqk9m9os7) 

<img src="Wandb_Augmentation_Panel.png">

## Findings
- Dataset: ImageNet-A, containing approximately 7,500 challenging images with significant domain shifts. The dataset poses a challenge due to:
  - Adversarial examples: Images are selected to be difficult for standard models.
  - Severe domain shifts: The test images differ greatly from training data, testing the model's adaptability.
- Backbone Model: CLIP (for zero-shot performance).
- Overall Accuracy:
  - Zero-Shot: 38.17%
  - Adapted: 47.55%
    
The implemented MEMO model showed significant improvements in top-1 accuracy on the ImageNet-A dataset, effectively handling domain shifts.

## Recommendations for Further Improvements
1. Explore Different Backbones: Test other models besides CLIP to potentially improve adaptation performance.
2. Enhance Augmentation Techniques: Experiment with advanced data augmentation methods to bolster robustness.
3. Parameter Tuning: Conduct hyperparameter optimization for better adaptation results.We 
4. Real-World Testing: Apply the model to other datasets with varying domain shifts to evaluate generalizability.

PS: This is the implementation of the paper for the Deep Learning Master Course at the University of Trento. [March 2024]
