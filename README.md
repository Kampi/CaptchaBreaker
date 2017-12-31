# Captcha-Solver

This is a private educational project for my *KI and Softcomputing* lecture at HSD in Germany.

Please write me an e-mail to DanielKampert@kampis-elektroecke.de if you have any questions

## Table of Contents
 - [About](#about)
 - [Requirements](#requirements)
 - [Install necessary packages](#install-necessary-packages)
 - [How does it work?](#how-does-it-work)
 - [History](#history)
 
### About
This is a machine learning project for my lecture at HSD in germany. This application can solve captchas by using a neural network with name "LeNet".
You train the network with some captchas from [Kaggle.com](https://www.kaggle.com/codingnirvana/captcha-images/data) and after the training the network will solve (similar) captchas from the internet.

### Requirements
- Python 3.6.2
- Visual Studio with Python tools

### Install necessary packages ##

```
$ python -m pip install numpy
$ python -m pip install opencv-python
$ python -m pip install scipy
$ python -m pip install scikit-learn
$ python -m pip install h5py
$ python -m pip install imutils
$ python -m pip install tensorflow
$ python -m pip install git+git://github.com/fchollet/keras.git
$ python -m pip install matplotlib
$ python -m pip install graphviz
$ python -m pip install pydot
```

You also need the `graphviz` executable for your OS.

## How does it work ##

1. Download the project
2. Install Python and the necessary packages
3. Navigate to the `src` directory of the project and open the project with visual studio or a python ide

## History

| Version   | Description                | Date       |
|:---------:|:--------------------------:|:----------:|
| 1.0       | First release              | 18.12.2017 | 
| 1.1       | Improve "Predict" method   |            | 