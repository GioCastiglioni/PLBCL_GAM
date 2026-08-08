# PLBCL_GAM

**Pixel Level Balanced Contrastive Learning & Graph Attention in the Latent Space**

This repository contains the source code for experiments regarding **Balanced Contrastive Learning (BCL)** applied at the pixel level, and **Graph Attention** processing in the latent space for downstream Earth Observation (EO) segmentation tasks.

## Features

### Datasets
- **PASTIS** (Time-series)
- **M-PV4GER-SEG** 
- **M-SA-CROP-TYPE**

### Encoders
- **ViT (Tiny / Small)**: Used without weights (trained from scratch).
- **ResNet18**: Pre-trained.
- **SSL4EO-MoCo**: ViT-Small Pre-trained foundation model.

### Losses
- **Pixel-Level Balanced Contrastive Learning (PL-BCL)**
- **Focal Loss**
- **Cross Entropy Loss**