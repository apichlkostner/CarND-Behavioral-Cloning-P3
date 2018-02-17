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
[image_plot_loss]: ./images/plot_loss.png "Plot loss"
[image_plot_loss_final]: ./images/plot_loss_final.png "Plot loss"
[image_architecture]: ./images/architecture.png "Architecture"

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

## Project files

The project includes the following files:
* model.py - creates and trains the model
* drive.py - connects to the simulator and drives the car
* model.h5 - contains the trained model
* video.mp4 - the video of the autonomous drive in track 1
* video_track2.mp4 - the video of the autonomous drive in track 2
* README.md - this document

## Usage of the model and the simulator

The python script drive.py can connect to the simulator and drive the car with

```sh
python drive.py model.h5
```


# Model Architecture and Training Strategy

## Base model architecture

The model is based on the work done by a team from NVidia [1].

A new model is created with the function `get_model(input_shape, horizon, small_net = True)` (model.py, line 192).

It consists of a convolutional neural network with kernel sizes of 3x3 and 5x5 and some fully connected layers.

As activation functions 'RelU' is used, regularization is done with 'Dropout' and 'batch normalization' is done.

Since a relatively large neural network is done some measures to prevent overfitting are implemented:

* Dropout between the layers
* Data augmentation
    * Using left and right camera
    * Virtual camera shift to the left and the right
* Aquisition of many samples
    * Aquisition of samples from different two tracks
    * Driving the tracks two times in both directions
    * Additional data aquisition for critical situations

To ensure that no overfitting occurs a train/validation split is done on the data with 80% training data and 20% validation data. Since the validation loss is lower than the training loss the measures against overfitting work.

![Plot loss][image_plot_loss]
![Plot loss][image_plot_loss_final]

The final test was done by driving the simulated car with the model and manually checking that the driving behavior is fine. A separate test set wouldn't help since a low loss don't necessarily results in a model that can drive the car correctly.

## Model parameter tuning

The neural network is based on a network architecture which was used by a team from NVidia to drive a car in real world. Since for this work the model has simpler conditions in the simulator the big fully connected layers were reduced in size and one layer was removed. Both, the original model and the reduced model were trained and both show the same good driving behavior.

The optimizer used was Adam so no extra parameter tuning was done.

## Getting the training data

Training data was produced by driving the simulator manually with a controller. Using the mouse was too difficult and using only the keyboard was not accurat enough.

Both tracks of the simulator were used, in both two loops in both directions were driven. Additional corrections from wrong car trajectories were recorded and special situations like curves or difficult to detect situations on the tracks were used to have better data.


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

![alt text][image_architecture]

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

References:
[1] http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
