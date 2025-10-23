# SARU-Net
A Shadow-Aware and Removal Unified Network for Remote Sensing Images with New Benchmarks

# Project README

This project provides instructions for using the code and datasets for shadow detection and removal. Below are the details for obtaining datasets, pretrained models, and running the code.

## 1. Obtain Pretrained Model Parameters

Download the pretrained model parameters from the following Baidu Netdisk link:
- **Link**: [https://pan.baidu.com/s/1myX_38t52SI7mtcTI6rHCw?pwd=3vqt](https://pan.baidu.com/s/1myX_38t52SI7mtcTI6rHCw?pwd=3vqt)
- **Extraction Code**: 3vqt

Download the following files:
- `DecoupleNet_D2.pth`
- `best_AISD_ckp.pth`
- `best_RSIISD_ckp.pth`

**Steps**:
1. Download the pretrained model files listed above.
2. Save the downloaded model files to the appropriate paths in the project (refer to the path configurations in the code).

## 2. Obtain RSISD Dataset

The proposed **RSISD Dataset** can be downloaded from the following Baidu Netdisk link:
- **Link**: [https://pan.baidu.com/s/1myX_38t52SI7mtcTI6rHCw?pwd=3vqt](https://pan.baidu.com/s/1myX_38t52SI7mtcTI6rHCw?pwd=3vqt)
- **Extraction Code**: 3vqt

**Steps**:
1. Download the RSISD dataset.
2. Extract the dataset and save it to the appropriate path in the project.

## 3. Obtain AISD Dataset

The **AISD Dataset** can be downloaded from the following GitHub page:
- **Link**: [https://github.com/RSrscoder/AISD](https://github.com/RSrscoder/AISD)

**Steps**:
1. Visit the GitHub page linked above.
2. Download the AISD dataset and extract it to the appropriate path in the project.

## 4. Test Shadow Detection and Removal

You can directly run the following command to test shadow detection and removal:

```bash
python demo.py
```

### Note
Before running, ensure that the file paths in demo.py are correctly configured to point to the pretrained models and datasets.

## 5. Train DBSFM Model
To train the DBSFM model, follow these steps:

Modify the relevant configurations in train.py (e.g., dataset paths, model parameters, etc.).
Run the following command:

```bash
python train.py
```
## 6. Test Shadow Detection Results
To test shadow detection results, follow these steps:

Modify the relevant configurations in test.py (e.g., dataset paths, model paths, etc.).
Run the following command:

```bash
python test.py
```
## 7. Quantitative Evaluation for Single Image Shadow Removal
To perform quantitative evaluation for single image shadow removal, follow these steps:

Modify the relevant configurations in quantity.py (e.g., image paths, model paths, etc.).
Run the following command:

```bash
python quantity.py
```
### Notes

Ensure all file paths are correctly configured to avoid errors during execution.
Before running the code, make sure the required Python environment and dependencies (e.g., PyTorch) are installed.
If issues arise, verify the integrity of the dataset and model files or refer to other documentation in the project.
