**Behavioral Cloning** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Figures/center.jpg "Center Image"
[image2]: ./Figures/left.jpg "Left Image"
[image3]: ./Figures/right.jpg "Right Image"
[image4]: ./Figures/center_cropped.jpg "Center Cropped Image"
[image5]: ./Figures/left_cropped.jpg "Center Cropped Image"
[image6]: ./Figures/right_cropped.jpg "Right Cropped Image"
[image7]: ./Figures/model_structure.png "Model Structure"
[image8]: ./Figures/Dropout3_0.4_0.1.png "Model error"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model_dropout.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model_dropout3.h5` containing a trained convolution neural network 
* `writeup.md` or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Model `model_dropout3.h5` can be obtained by running 
```sh
python model_dropout.py
```

Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_dropout3.h5
```

#### 3. Submission code is usable and readable

The `model_dropout.py` file contains the code for data proprecessing, data augmentation, model architecture building, model training, model validation, and model saving with brief comments.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the sequential API to build a model contains one normalization layer, one dropping layer, five convolution layers, one flatten layer, four fully connected layers with two of them have dropout (`model_dropout.py` lines 82-113). Nonliearity was introduced by using RELU activation function. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`model_dropout.py` lines 107， lines 111). Data were shuffled and randomly split into training and validation dataset, and different dropout rates were tested to reduce the validation error.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model_dropout.py` line 118).

#### 4. Appropriate training data

I used the provided training data including left, center, and right camera images, and also augmented data by flipping. An ideal training data set should inlude at least one lap and senarios where car recovering to the road center from driving off the lane.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with different model layers, adding nonlinearity and dropout.

I started with the default model architecture (`model_dropput.py` line 88 - 95) which was not complicated enough to obtain a statisfying model. In addition, the validation error was much higher than training error indicating an overfitting issue. 

I next tried a more complicated model architecture based upon NIVIDA's model without dropout, the validatoin error reduced a bit. But the simulator showed the car didn't have smooth performance at wide turns. Therefore, I further added two dropout layers and experimented different dropout rates.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture.  

![alt text][image7]

#### 3. Creation of the Training Set & Training Process

Training data were created based upon the provided dataset. Data was preprocessed with color transformation from BGR to RGB, left/right camera image measurement correction, augmentation, normalization and cropping. I finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. A model which could drive the car at 30 mph staying in the lane center could be obtained by using 3 epochs. The validation error also showed 3 epochs were enough for the model training.

Here are examples of original and cropped images:   
Center image example

![center image][image1]

Left image example

![center image][image2]

Right image example

![center image][image3]

Center cropped image example

![center image][image4]

Left cropped image example

![center image][image5]

Right cropped image example

![center image][image6]

Here is the training process:
![alt text][image8]


### Video Recording

#### 1. Provide a link to your final video output.  
Here is the link to the project output video:  
[![video image](https://img.youtube.com/vi/6sbUE5AuaKw/0.jpg)](https://youtu.be/6sbUE5AuaKw)  



