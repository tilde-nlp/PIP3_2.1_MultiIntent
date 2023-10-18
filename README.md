# PIP3_2.1_MultiIntent
Scripts for training the Multi-intent detection model

## Usage
To train the Multi-intent detection model the train and test data must be prepared. The data are included in the two column TAB separated file. Each line contains an utterance example in the first column and the intent identifier in the second column.

Training is performed with the script [run_training.py](run_training.py). All infomation used for the training is specified in the [config.ini](config.ini) file.

Parameters in the *config.ini* file:

- datadir: directory where different statistics will be stored
- resdir: result directory
- input_file_train: file with training data
- input_file_test: file with test data
- dsName: name of the dataset
- lang: language of the data
- xval: number of x-validation steps; testing is done on test data if xval=0
- multi: if true detects multiple intents per example
- verNr: model version - 0, 1, or 2
- epochs: number of epochs model must be training

Folder *Classifier* contains class for the classifier and functions for training and validation.

Folder *DataPreprocessing* contains functions for data preprocessing.

Folder *DataVectorizing* contains class for connecting to FastText vectorizer.

Folder *ResultStats* contains functions for statistics calculation.

Folder *VectorizerService* contains Docker container code for starting fastText vectorizer service.

## Architecture
We have implemented 3 different multi-intent detection models

### Version 0 - Many Single-Intent Models

Each model is a classifier with two output classes that predicts if there is a particular intent or if this intent is absent.

![0.png](0.png)

### Version 1 - Single Multi-Intent Model with Many Dense layers

All models are joined in a single model with a common convolution layer and *n* dense layers with 2 units for each intent. Like with the previous
approach, each of the dense layers has 2 output classes - either the intent exists or not. Outputs of the dense layers are concatenated, thus the output size of the concatenation
layer is 2 × *n*. The next Lambda layer throws out every other output leaving only those who signalize the existence of the intent.

![1.png](1.png)

### Version 2 - Single Multi-Intent Model with a Common Dense Layer

For this architecture, we use a common dense layer for all intents, but instead of the softmax activation function we use the sigmoid activation function that allows us to
detect if an intent is in an utterance regardless of other intents.

![2.png](2.png)

## Results

Version 0 approach can be slow if there are many intents in the data as it is necessary to train (and support) *n* individual classifiers. Version 1 is faster than having to train individual models for each intent. However, the middle layer consists of many independent layers that are relatively slow.  Experiments showed that the best-performing architecture for the multi-intent classification was Version 2 model. It allowed to achieve the highest accuracy scores with acceptable training times.

In single-intent models, we assumed that the correct intent is the one with the top confidence value that is larger than 0.5. As we have replaced the softmax activation with the
sigmoid activation in our final architecture (Version 2), the sum of confidence scores for all intents does not have to be 1.0. Therefore, we need to establish threshold value t ∈ [0, 1]. If the confidence value for an intent returned by the model is above the threshold, we can accept that there is such an intent in the example. The threshold value for multi-intent models can be induced from cross-validation results for each individual dataset. For our sample dataset, the best F1 value of 0.955 is achieved with the threshold set at 0.12.

![results.png](results.png)

## Acknowledgment
This prototype is created in activity 2.1 of the project "AI Assistant for Multilingual Meeting Management" (No. of the Contract/Agreement: 1.1.1.1/19/A/082).

@Tilde, 2022
