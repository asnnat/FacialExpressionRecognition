# FacialExpressionRecognition
CN240 - DATA SCIENCE FOR SIGNAL PROCESSING

## About The Project
FacialExpressionRecognition is a project that predict an emotion from facial expression.
The emotions are
* ANGER
* CONTEMPT
* DISGUST
* FEAR
* HAPPINESS
* NEUTRAL
* SADNESS
* SURPRISE

### Built With
Machine learning

## Getting Started
### Prerequisites
You have to install software before using the project.

1. Download [Python](https://www.python.org/downloads/)
2. Install [Visual Studio Code](https://code.visualstudio.com/download)

### Installation
1. Clone the repo
    ```sh
    git clone https://github.com/asnnat/FacialExpressionRecognition.git project
    ```
2. Change directory to the project
    ```sh
    cd project
    ```
3. Open the directory with Visual Studio Code
    ```sh
    code .
    ```

## Roadmap
### Gathering data from Kaggle
* [Emotion Detection From Facial Expressions](https://www.kaggle.com/c/emotion-detection-from-facial-expressions/overview)
* [Corrective re-annotation of FER - CK+ - KDEF](https://www.kaggle.com/datasets/sudarshanvaidya/corrective-reannotation-of-fer-ck-kdef)

### Create all_data.csv from dataset
Use the `pre_processing.ipynb` to create `preprocessing_data.csv` metadata file

### Image processing
Use the `pre_processing.ipynb` to preprocess image by
* Resize
* Convert color
* Apply CLAHE

### Feature engineering
Use the `feature_engineering.ipynb` for
* Detect eyes, nose and lip with cv2 CascadeClassifier
* Plot 68 face landmarks with dlib
* Create 23 features from distance and angle between face landmarks
* Generate Correlation Heat Maps in Seaborn then filter only features that have value more than 0.90
* Feature important
* Wraper
* Embedded

### Train models
Use the `cross_validate.ipynb` `cross_validate_emb.ipynb` `cross_validate.ipynb_wrp` for train models with 5-fold crossvalidation by using
* SVM linear kernel
* SVM poly kernel
* SVM rbf kernel
* Logistic regression
* Random forest
* Guassian naive bayes

### Evaluate models
Use the `evaluate.ipynb` for
* Evaluate models
* Create confusion matrixs
* Get classification report

