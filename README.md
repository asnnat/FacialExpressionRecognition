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
## Usage

## Roadmap
### Gathering data from Kaggle
* [Emotion Detection From Facial Expressions](https://www.kaggle.com/c/emotion-detection-from-facial-expressions/overview)
* [Corrective re-annotation of FER - CK+ - KDEF](https://www.kaggle.com/datasets/sudarshanvaidya/corrective-reannotation-of-fer-ck-kdef)

### Create all_data.csv from dataset
Use the `pre_processing.ipynb` to create metadata.

### Image processing
Use the `pre_processing.ipynb` to preprocess image.
* Resize
* Convert color
* Apply CLAHE

###