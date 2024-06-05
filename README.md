# SoftComputing_PredefinedProject2

**Course:** *Soft Computing*  
**University:** _Faculty of Technical Sciences, Novi Sad_  
**Field of Study:** **Software Engineering and Information Technologies**  
**Project:** *Predefined project 2 - Weather Classification Challenge*  
**Student:** __Marko Suhanov__  



## Project description
This project aims to classify weather conditions into four classes: **sunrise**, **cloudy**, **shine**, and **rain**. The dataset is provided for both training and testing purposes. The classifier will be trained on the training dataset and then used to classify images from the test dataset with the highest possible accuracy.

## Task
The following link provides a dataset for the problem of classifying weather conditions into 4 classes (**sunrise**, **cloudy**, **shine**, and **rain**).
- The training dataset is located in the `train/` directory.
- The file **train_labels.csv** contains a list of images in the training dataset with their corresponding classes.
- The test dataset is located in the `test/` directory.
- The file **test_labels.csv** contains a list of images in the test dataset with their corresponding classes.
- Create a classifier that will learn to classify images from the test dataset based on the images in the training dataset with the highest possible accuracy.
- To achieve a maximum of 40 points (grade 7), it is necessary to achieve an accuracy > 70%.

## Requirements
- Python 3.x
- TensorFlow (Keras)
- Other dependencies listed in **requirements.txt**

## Installation
### Create Environment 
```
python -m venv venv
```

### Activate Environment 
**Linux / Mac**
```
source ./venv/bin/activate
```

**Windows**
```
./venv/Scripts/activate.bat
```

### Install Requirements 
```
pip install -r requirements.txt
```

## Usage
To train the classifier and make predictions, run the following command:
```
python main.py
```

