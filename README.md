## Dataset information
The dataset is a collection of retrospective pre-operative brain magnetic resonance imaging (MRI) scans, clinically acquired from six diagnostic centers in Nigeria. The scans are from 146 patients who have brain MRIs indicating central nervous system neoplasms, diffuse glioma, low-grade glioma, or glioblastoma/high-grade glioma. The brain scans were multiparametric MR images (mpMRI), specifically T1, T1 CE, T2, and T2 FLAIR,  acquired on 1.5T MRI between January 2010 and December 2022. 

Scans were obtained from different scanners using each center’s acquisition protocol. Each scan was de-identified and de-faced to remove personal identifiers and presented in their original state with respect to resolution and orientation. To ensure uniformity across scans and modalities, a standardized pre-processing protocol was applied to adjust the image dimensions and voxel sizes. The scans were extracted from the PACs as DICOM files and converted to the Neuroimaging Informatics Technology Initiative (NlfTI) file format to facilitate computational analysis, following the well-accepted pre-processing protocol of the International Brain Tumour Segmentation (BraTS) challenge. All scans were subjected to sanity checks to confirm the presence of all required sequences. Specifically, all mpMRI volumes were reoriented to the left posterior-superior (LPS) coordinate system, and the T1 CE scan of each patient was rigidly (6 degrees of freedom) registered and resampled to an isotropic resolution of 1 mm3 based on a common anatomical atlas, namely SRI. The remaining scans (i.e., T1, T2, FLAIR) of each patient were then rigidly co-registered to this resampled T1 CE scan by first obtaining the rigid transformation matrix to T1 CE, then combining with the transformation matrix from T1 CE to the SRI atlas, and resampling. The N4 bias field correction was applied in all scans to correct for intensity non-uniformities caused by the inhomogeneity of the scanner's magnetic field during image acquisition to facilitate an improved registration of all scans to the common anatomical atlas. Brain extraction was also performed using a standard process for  skull-stripping to remove all non-brain tissue (including neck, fat, eyeballs, and skull) from the image and create a brain mask to  enable further computational analyses.

More details: https://www.cancerimagingarchive.net/collection/brats-africa/

## Project description
I completed the task of segmenting three areas of the brain:
- Whole Tumore = Oedema + Enhancing Tumore + Necrotic Tumore
- Enhancing Tumore
- Tumore Core = Enhancing Tumore + Necrotic Tumore

109 patients were identified in the training sample and 37 in the validation sample.

To solve the problem, I trained the ResidualUNet3d model for segmentation.

I used Dice as a model quality metric.

I used a composite loss function consisting of FocalLoss, BoundaryLoss, RegionDiceLoss, and losses by region Enachancing Tumore and Tumore Core.

## Requirements
pip install -r requirements.txt

## Run
Open brats_africa_ResidualINet3d.ipynb

## Model Weights
Download weights from: <https://huggingface.co/EvgeniyEV/ResidualUNet3d/blob/main/best_model_.pth>
