# Web Highlighing Tool for Business Reports
## Introduction
This is the implementation refers to the paper *Towards Bankruptcy Prediction: Deep Sentiment Mining to Detect Financial 
Distress from Business Management Reports*. 

Runs the model with the german word2vec of Andreas Müller (devmount, https://github.com/devmount). Please cite the 
author when using the data.

## Requirements
Code is written in Python 3.6 and requires 

* Keras (2.1) with Theano or Tensorflow
* Django (2.0)
* NLTK (3.2) with German and English language packs
* scikit-learn (0.19)
 
## Configuration
### Set paths
Open `root/webhighlighting/settings.py` and set the correct paths where you find the `# TODO` comments.
 
### Get word embeddings
If you want to train english models instead of german models, you have to upload the english word2vec dataset to the 
directory `root/trainmodel/data/` and call the file `english_300dim.model`. We recommend the google word2vec dataset
by Tomas Mikolov et al. (https://code.google.com/archive/p/word2vec/).

## Use highlighting with another model
Check the variable `mp_` in `root/highlight/__init__.py` and set the correct path to your model. Leave the extension 
out. This tool will expect in the set directory files with the extensions `yourfilename.patterns`, 
`yourfilename.model` and `yourfilename.data`. These files are generated in the training process of this tool. 

## Run the tool
You start the server by running `python3 manage.py runserver` in your console. In the beginning django will load the
model that is used for highlighting, this may take a while (probably up to 2h, but this is only needed initially). 
After the model is loaded, you will see a URL in the console (most likely http://127.0.0.1:8000/). Open this URL in your 
browser to access the tool. 

