# Car and Plane Detection using CNN

## Introduction

![car_planeproject](https://user-images.githubusercontent.com/40186859/120678171-8dae4600-c4b7-11eb-80bf-2d05c96a1740.png)


This project mainly focused on car plane detection based on Convolution Neural Network. We are using Python 3.7.3 and NumPy to build this program. But, you can use any version of Python to run this code.

The main aim of this tutorial is to improve our capacity to analyze the operating process and various CNN architecture based on detecting the car and the plane. Also to obtain knowledge on the basic parameter and hyperparameter that make up a complete program. 

## Steps to build a complete car plane detection model

* The initial step is to collect the data for car and plane
* Divide the dataset into train, test, and validate dataset
* Train the CNN architecture to detect the car and the plane from our dataset which we have created.
* The final step is to evaluate the result


## Working of CNN

We have introduced our CNN architecture for the classification and extraction of features from the dataset which we have created earlier.

CNN is also known as a multilayered neural network that is applied for image processing problems like the text in images, image recognition, self-driving cars, and powering vision in robots. CNN is a network of the neuron and has a three-layer which is named as a convolutional layer, padding layer, and third one ins fully connected layer. Our network is made up of 11 layers which do not involve the input layers because the input layer brings in a representation of the RGB color where each color is handled independently. 

In our CNN architecture, the first two convolution layer is applied to an image in the layer by 16 of 3*3 filters. The third and fourth convolution layer is applied to an image in the layer by 32 of 3*3 filters.

The sub-layer of non-linear transformation uses the ReLu activation function because if the neuron activates the gradient is still strong which is equal to 1. A 2*2 filter is applied to the image by the max-pooling sublayer which results in the image size being reduced to half. In this point, each color channel is defined by 32*32 array which was extracts from 64 features of the convolutional network. 

The eighth layer of our CNN architecture is known as the flatten layer to convert the multi-dimensional matrix of features into a one-dimensional array that can be fed into a neural network classifier that is fully connected and the one-dimensional array with size 4800 is used as the output of the flatten layer. The ReLu activation function with a fully connected artificial neural network (ANN) is used in the ninth layer which maps input values of 4800 to output values of 64. The dropout layer is the tenth layer. Thus, to overcome the issue of overfitting, fifty percent of the input values that come with layers is reduced to zero. The last eleventh layer is an ANN which is fully connected to maps 64 inputs values of 2 class labels with a sigmoid activation function.

Firstly, we use the data in the training set to train the convolution network to find suitable weights of filters in the three convolutional sub-layers and the weights which offer the two fully connected layers with a minimal error. Next, we use the data in the validation set to test the convolution network to get validation error and cross-entropy loss. Within the same process, we repeat the convolution network training until we reach up to 10 epochs. 

## Result and Analysis

We have plotted different graphs like iteration obtained graphs, graph based on loss and accuracy, and confusion matrix. Each graph shows the performance of our model. 

While training the convolution network we have observed that loss is decreasing in each epoch. And also, shows that there is a minimum error while training our data. 

![result-and-analysis](https://user-images.githubusercontent.com/40186859/120801044-63fd2980-c560-11eb-9907-72d610287ccb.png)

Next, we have evaluated the graph based on model accuracy and model loss which shows the performance of our model on training and validation set, and each interaction we have observed better results.

![model_loss_accuracy](https://user-images.githubusercontent.com/40186859/120801191-93ac3180-c560-11eb-8728-e69848585f90.png)

At last, we have created a confusion matrix to evaluate the performance of our model. It provides us insight to calculate the mistake provided by a classifier. 

![confusion matrix](https://user-images.githubusercontent.com/40186859/120801486-f0a7e780-c560-11eb-8a8a-9926166b74ae.png)


