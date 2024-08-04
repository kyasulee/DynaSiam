# A Two-stage Image Enhancement and Dynamic Feature Aggregation Framework for Gastroscopy Image Segmentation

## Abstract
Accurate and reliable automatic segmentation of lesion areas in gastroscopy images can assist endoscopists in making diagnoses and reduce the possibility of missed or incorrect diagnoses. This paper presents a two-stage framework for segmenting gastroscopy images, which aims to improve the accuracy of medical image segmentation tasks using limited datasets. The proposed framework consists of two stages: the Image Enhancement Stage and the Lesion Segmentation Stage. First, in the Image Enhancement Stage, an image enhancement solution called TDC-Enhance is proposed to enrich the original small-scale gastroscopy image dataset. This solution performs Texture Enhancement, Detail Enhancement, and Color Enhancement on the original images. Then, in the Lesion Segmentation Stage, a multi-path automatic segmentation network for gastroscopy images, named DynaSiam, is introduced. DynaSiam comprises a Dependent Encoder, a Shared Encoder, and a Fusion Decoder. It learns feature information related to the lesion region by encoding the different enhanced images obtained in the Image Enhancement Stage as inputs to the multi-path network. Additionally, a Dynamic Feature Interaction (DFI) block is designed to capture and learn deeper image information, thereby improving the segmentation performance of the model. The experimental results show that the proposed method achieves a 90.80\% mIoU, 92.71\% Dice coefficient and 96.31\% Accuracy. Other performance metrics also indicate the best performance, suggesting that the proposed model has significant potential for clinical analysis and diagnosis.

## Running the project
When you have completed the configuration and prepared the desired dataset, you can ```python train.py``` 

If you have any questions, please contact me at 846483478@qq.com


