## Dataset information
The dataset is a collection of retrospective pre-operative brain magnetic resonance imaging (MRI) scans, clinically acquired from six diagnostic centers in Nigeria. The scans are from 146 patients who have brain MRIs indicating central nervous system neoplasms, diffuse glioma, low-grade glioma, or glioblastoma/high-grade glioma. The brain scans were multiparametric MR images (mpMRI), specifically T1, T1 CE, T2, and T2 FLAIR,  acquired on 1.5T MRI between January 2010 and December 2022. 

Scans were obtained from different scanners using each center’s acquisition protocol. Each scan was de-identified and de-faced to remove personal identifiers and presented in their original state with respect to resolution and orientation. To ensure uniformity across scans and modalities, a standardized pre-processing protocol was applied to adjust the image dimensions and voxel sizes. The scans were extracted from the PACs as DICOM files and converted to the Neuroimaging Informatics Technology Initiative (NIfTI) file format to facilitate computational analysis, following the well-accepted pre-processing protocol of the International Brain Tumour Segmentation (BraTS) challenge. All scans were subjected to sanity checks to confirm the presence of all required sequences. Specifically, all mpMRI volumes were reoriented to the left posterior-superior (LPS) coordinate system, and the T1 CE scan of each patient was rigidly (6 degrees of freedom) registered and resampled to an isotropic resolution of 1 mm3 based on a common anatomical atlas, namely SRI. The remaining scans (i.e., T1, T2, FLAIR) of each patient were then rigidly co-registered to this resampled T1 CE scan by first obtaining the rigid transformation matrix to T1 CE, then combining with the transformation matrix from T1 CE to the SRI atlas, and resampling. The N4 bias field correction was applied in all scans to correct for intensity non-uniformities caused by the inhomogeneity of the scanner's magnetic field during image acquisition to facilitate an improved registration of all scans to the common anatomical atlas. Brain extraction was also performed using a standard process for  skull-stripping to remove all non-brain tissue (including neck, fat, eyeballs, and skull) from the image and create a brain mask to  enable further computational analyses.

More details: https://www.cancerimagingarchive.net/collection/brats-africa/

## Project description
The task is to segment three tumor sub-regions:
- Whole Tumor (WT) = Edema + Enhancing Tumor + Necrotic Core
- Tumor Core (TC) = Enhancing Tumor + Necrotic Core
- Enhancing Tumor (ET)

The dataset was split into 109 training patients and 37 validation patients.

**Model:** Residual 3D U-Net (ResidualUNet3d).

**Loss:** a composite loss combining Focal Loss, Boundary Loss, and
Region-based Dice Loss (with additional region terms for Enhancing Tumor
and Tumor Core).

**Metric:** Dice score.

Mean validation Dice (TTA) = **0.8095**
| Region | Dice  |
|--------|-------|
| WT     | 0.836 |
| TC     | 0.788 |
| ET     | 0.806 |

## Installation & Local Run

### 1. Clone the repository
```bash
git clone https://github.com/EvgeniyKaduk/BraTS_Africa_segmentation.git
cd BraTS_Africa_segmentation
```

### 2. Create environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If you receive the error "Script execution is disabled on this system,"
execute the following command once in PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then repeat the activation: `venv\Scripts\activate`

### 3. Download model weights & demo data
```bash
python download_assets.py
```
Or download manually:
- **Weights** → https://huggingface.co/EvgeniyEV/ResidualUNet3d/blob/main/best_model_.pth
  Place into `models/best_model_.pth`
- **Demo data** → https://huggingface.co/datasets/EvgeniyEV/brats-africa-demo/tree/main
  Extract into `demo_data/`

Demo subject folder must contain 4 modalities (and optionally `seg`):
```
demo_data/demo_subject_001/
├── *-t1n.nii.gz
├── *-t1c.nii.gz
├── *-t2w.nii.gz
├── *-t2f.nii.gz
└── *-seg.nii.gz   # optional (Ground Truth)
```

### 4. Run demo inference
All patients, save to files + show windows:
```bash
python demo_inference.py --data_path demo_data --model_path models/best_model_.pth
```
All patients, only save as PNG (no windows):
```bash
python demo_inference.py --data_path demo_data --model_path models/best_model_.pth --no_show
```
After this, all 3 images will be saved in the outputs/ folder.
The images will show: **Ground Truth | Prediction | Errors**.

Specific patient (3rd):
```bash
python demo_inference.py --data_path demo_data --model_path models/best_model_.pth --subject_idx 2
```

### 5. (Optional) Train from scratch
```bash
python train.py --data_path dataset/BraTS-Africa --epochs 40
```