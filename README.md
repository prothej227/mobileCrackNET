# MobileCrackNet
## Description

A crack image classifier trained on pre-trained MobileNet. The data is trained on [Data Mendeley Concrete Crack Dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/1). The model has an average validation accuracy of 0.9982.

## Requirements

- numpy==1.19.5
- tensorflow==2.11.0
- tensorflow_gpu==2.5.0

## Training Guide

Refer to the Jupyter Notebook file [Training.ipynb](Training.ipynb) for the training guide.

## Usage

### Prerequisites

Make sure you have the required Python packages installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```
### Prediction
To predict the class and score for an image, run the following command:

```bash
python predict.py -m model/weights.h5 -i test/00007.jpg
```
Replace model/weights.h5 with the path to your trained model and test/00007.jpg with the path to the image you want to predict.


## Authors
- prothej227/Journel Cabrillos