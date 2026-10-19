\# VISTA-XL

\### Vision xLSTM-Enhanced Swin-UNet for Efficient Medical Image Segmentation



VISTA-XL is a research project exploring the integration of an xLSTM memory block into a Swin-Tiny encoder and U-Net-style decoder for binary skin-lesion segmentation.



\## Overview



The project investigates whether adding matrix-LSTM (mLSTM) feature processing to a hierarchical vision-transformer encoder can improve segmentation performance.



\*\*Proposed architecture:\*\*

\- \*\*Encoder:\*\* Pretrained Swin-Tiny

\- \*\*Feature processing:\*\* mLSTM block applied to the deepest spatial feature map

\- \*\*Decoder:\*\* Convolutional decoder with multi-scale skip connections

\- \*\*Output:\*\* Binary segmentation logits



\## Dataset



The current experimental setup uses:

\- 2,000 training images

\- 150 validation images

\- RGB images and binary lesion masks

\- Input resolution: 224 × 224 pixels



The dataset's provenance, image-mask alignment, and independent test-set availability must be verified before final research claims are made.



\## Baseline Models



Two baseline experiments have been completed:

\- U-Net

\- Swin-Tiny encoder with a convolutional decoder



VISTA-XL training is currently in progress. Its final performance has not yet been established.



\## Repository Structure



```text

VISTA-XL/

├── configs/

│   └── vista\_xl.yaml

├── docs/

├── notebooks/

├── scripts/

├── src/

│   ├── models/

│   │   └── vista\_xl.py

│   ├── losses.py

│   └── metrics.py

├── checkpoints/

├── results/

├── Dataset/

├── unet-project.ipynb

└── vista\_xl\_implementation\_plan.md

```



\## Experimental Setup



The current Kaggle experiment uses:

\- Image size: 224 × 224

\- Batch size: 8

\- Optimizer: AdamW

\- Learning rate: 0.0001

\- Weight decay: 0.00001

\- Training duration: 10 epochs

\- Loss: Binary cross-entropy plus soft Dice loss

\- Validation metrics: Dice coefficient and Intersection over Union (IoU)



The pretrained Swin encoder uses ImageNet normalization. Training checkpoints and CSV history are saved separately from the baseline artifacts.



\## Current Status



\- \[x] U-Net baseline trained

\- \[x] Swin baseline trained

\- \[x] mLSTM package installed and independently smoke-tested

\- \[x] VISTA-XL architecture implemented and smoke-tested in Kaggle

\- \[x] Real-batch forward and backward test passed

\- \[ ] Complete VISTA-XL training

\- \[ ] Verify checkpoint and training history

\- \[ ] Audit suspicious image-mask pairs and dataset provenance

\- \[ ] Perform controlled model comparison

\- \[ ] Run ablation and repeatability experiments

\- \[ ] Complete research report and final evaluation



\## Research Caution



A successful architecture test does not establish segmentation superiority. Final conclusions require controlled comparisons, an appropriate independent test strategy, documented preprocessing, and reproducible evaluation.



\## Reproducibility



The model definition, loss, metrics, and experiment configuration are being organized into reusable Python modules. The current YAML file documents the intended settings; a configuration-driven training entry point is still to be implemented.



\## License and Data



Dataset licensing, source attribution, and repository licensing should be documented before public release.
