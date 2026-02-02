# RefinedStyle:Reconstruct the style image using AdaIN for style transfer
---
## Introduction
This code is provided for **RefinedStyle**. RefinedStyle enhances semantic integration by injecting textual context into the key and value components of style features. Furthermore, it reconstructs the style features via AdaIN to enable their smooth transfer to the textual attention. Experimental results further confirm that RefinedStyle achieves strong text-image alignment while maintaining consistent style transfer.
---
## Framework

## Results

### 1. Download
```bash
# git clone this repository
git clone https://github.com/math-ddup/StyleBoost.git
cd StyleBoost

# download the models
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
```

### 2. Install the environment
```bash
conda create -n refinedstyle python=3.10 -y
conda activate refinedstyle

pip install -r requirements.txt
```

### 3. Run
```bash
python infer_style.py
```

## Acknowledgements
Our work is mainly based on the following projects:
- [InstantStyle](https://github.com/instantX-research/InstantStyle.git)
