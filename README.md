# visual-textual-analysis
## Deep Learning techniques for Visual and Textual Analysis of images in the Retail environment

This project has been included in the paper "Robotic Retail Surveying by Deep Learning Visual and
Textual Data" accepted in Robotics and Automation Society, 2018. Tested on:

- Ubuntu 16.04
- Python 2.7
- NVIDIA GeForce GTX 960M (4 GB GDDR5)
- CUDA v8.0
- cuDNN v6.0

You can test these scripts on the following dataset:
- [SMART Dataset](http://vrai.dii.univpm.it/content/smart-dataset)
  - "SOOS" Example: 
  
    <img src="https://github.com/martinimass/visual-textual-analysis/blob/master/dataset/SOOS.jpg" width="200" height="200" />
  - "Normal" Example: 
  
    <img src="https://github.com/martinimass/visual-textual-analysis/blob/master/dataset/Normal.jpg" width="200" height="200" />
  - "Promotion" Example: 
  
    <img src="https://github.com/martinimass/visual-textual-analysis/blob/master/dataset/Promotion.jpg" width="200" height="200" />

## Python Environment Setup:
You should install the following python dependencies:
- sklearn
- PIL
- skimage
- concurrent

## Neural Networks Environment Setup:
Each phase of the project shold have its own environment. We recommend to install different virtual environments.
  
  ### VISUAL FEATURE EXTRACTION:
  - Install [CAFFE Framework](https://github.com/BVLC/caffe) and all his dependencies
  
  ### TEXTUAL FEATURE EXTRACTION:
  - Download and install the project [TextBoxes](https://github.com/mathDR/reading-text-in-the-wild) with all dependencies
  - Download and install the project [reading-text-in-the-wild](https://github.com/mathDR/reading-text-in-the-wild) with all dependencies
  - Download and install the project for Text Classification

## Dataset:
- Download the dataset files in the `/dataset` folder.

## Run:
Move to the `/bin` folder and run the scripts from there.
### VISUAL FEATURE EXTRACTOR:
#### - FINE-TUNING:
  ```
  bash visual/visual-finetuning.sh NETWORK 
  ```
  NETWORK = VGG16 | AlexNet | CaffeNet | GoogLeNet | ResNet50 | ResNet101 | ResNet152
#### - TEST:
  ```
  python visual/visual-testmodel.py NETWORK N_ITER 
  ```
  N_ITER = number of iterations for the best trained weights of the network;
#### - FEATURE EXTRACTION:
  ```
  python visual/visual-feature-extractor.py NETWORK N_ITER
  ```

### TEXTUAL FEATURE EXTRACTOR:
#### OCR:
  1) Text Detection
  ```
  python textual/text1_detection
  ```
  remember to change the path regardind the dataset and TextBoxes (.caffemodel and .prototxt files);
  
  2) Text Arrangement
  ```
  python textual/text2_arrangement.py
  ```
  3) Text Recognition
  ```
  bash textual/text3_recognition.sh
  ```
  remember to change the path regarding reading-text-in-the-wild (in text3_recognition.sh, batch_charnet.py and batch_dictnet.py);
  
  4) Text Assembly
  ```
  python textual/text4_assembly.py
  ```
  5) Text Encoding
  ```
  python textual/text5a_clean_ocr.py
  ```
  ```
  python textual/text5b_encoding.py
  ```
#### TEXTUAL FEATURE EXTRACTOR:
##### - FINE-TUNING:
  ```
  bash textual/text_finetuning.sh TXT_ARCH
  ```
  TXT_ARCH = [char2|dict2];
  
  remember to change the path regarding caffe-txt (.caffemodel and .prototxt files);

##### - TEST:
  ```
  bash textual/text_testmodel.sh
  ```
  remember to change the path regarding caffe-txt and number of iterations for the best trained weights of the networks (char2|dict2);
##### - FEATURE EXTRACTION:
  ```
  bash textual/text_feature_extractor.sh
  ```
  remember to change the path regarding caffe-txt;

### OVERALL:
#### - UNION OF VISUAL AND TEXTUAL FEATURES:
  ```
  bash create_final_dataset.sh VIS_CNN VIS_LAYER TXT_MODEL TXT_LAYER
  ```
  VIS_CNN = CNN model used for Visual Feature Extractor;
  
  VIS_LAYER = CNN layer used for Textual Feature Extractor;
  
  TXT_MODEL = CNN architecture used for Textual Feature Extractor;
  
  TXT_LAYER = CNN layer used for Textual Feature Extractor;
  
#### - MACHINE LEARNING CLASSIFIERS:  
  ```
  bash final_classifier.sh VIS_CNN VIS_LAYER TXT_MODEL TXT_LAYER N_THREADS
  ```
  N_THREADS = Number of parallel threads


