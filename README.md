# **Behavioral Cloning** 

# DRAFT DRAFT DRAFT

This document is the writeup for an exercise of an online course.


---

## Summary

The following steps were done:
* Images and steering angles were obtained by driving in a racing simulator on two tracks
    * Good trajectories
    * Corrections from bad trajectories
* With data augmentation more samples were generated:
    * Views from a left and a right camera were used
    * Between the camera views additional image were created by virtually shifting the camera
* Convolutional neural networks were trained with the data and connected to the simulator to drive the car automatically
* The driving behavior was analyzed and critical situations were found
* Extra samples of critical situations were created by driving the simulator manually
* A good working model was used to drive the simulator and videos of the result were recorded


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./images/simulator_manual.jpg "Simulator manual"
[image3]: ./images/simulator_autonomous.jpg "Simulator autonomous"
[image4]: ./images/difficult_situation_track1_01.jpg "Difficult situation"
[image5]: ./images/difficult_situation_track1_02.jpg "Difficult situation"
[image6]: ./images/difficult_situation_track1_03.jpg "Difficult situation"
[image7]: ./images/difficult_situation_track2_01.jpg "Difficult situation"
[image12]: ./images/difficult_situation_track2_02.jpg "Difficult situation"
[image13]: ./images/difficult_situation_track2_03.jpg "Difficult situation"
[image14]: ./images/difficult_situation_track2_04.jpg "Difficult situation"
[image8]: ./images/augmentation_example_01.jpg "Augmentation example"
[image9]: ./images/augmentation_example_02.jpg "Augmentation example"
[image10]: ./images/augmentation_example_03.jpg "Augmentation: flip"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image14]
![alt text][image12]
![alt text][image13]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
