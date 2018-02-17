[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[simulator_manual]: ./images/simulator_manual.jpg "Simulator manual"
[simulator_autonomous]: ./images/simulator_autonomous.jpg "Simulator autonomous"
[recovery]: ./images/recovery.jpg "Recovery"
[difficult01]: ./images/difficult_situation_track1_01.jpg "Difficult situation"
[difficult02]: ./images/difficult_situation_track1_02.jpg "Difficult situation"
[difficult03]: ./images/difficult_situation_track1_03.jpg "Difficult situation"
[difficult04]: ./images/difficult_situation_track2_01.jpg "Difficult situation"
[difficult05]: ./images/difficult_situation_track2_02.jpg "Difficult situation"
[difficult06]: ./images/difficult_situation_track2_03.jpg "Difficult situation"
[difficult07]: ./images/difficult_situation_track2_04.jpg "Difficult situation"
[augmentation01]: ./images/augmentation_example_01.jpg "Augmentation example"
[augmentation02]: ./images/augmentation_example_02.jpg "Augmentation example"
[augmentation03]: ./images/augmentation_example_03.jpg "Augmentation: flip"
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
* ConvNeurNet.py - the functions to create the neural networks
* ImageGenerator - the image generator with augmentation
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

For track two the target speed should be adjusted to 15 Mph since it has some sharp curves.

The autonomous mode looks like

![Autonomous mode][simulator_autonomous]

# Model Architecture and Training Strategy

## Base model architecture

The models were based on the work done by a team from NVidia [1] and the Inception module [2].

A new model is created with the function `get_model`.

It consists of a convolutional neural network with kernel sizes of 3x3 and 5x5 and some fully connected layers.

As activation functions 'elu' is used, regularization is done with 'Dropout' and 'batch normalization' is done.

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

![Plot loss][image_plot_loss_final]

The final test was done by driving the simulated car with the model and manually checking that the driving behavior is fine. A separate test set wouldn't help since a low loss don't necessarily results in a model that can drive the car correctly.

## Model parameter tuning

Networks with different depth of the convolutional and fully conntected layers were tested as well as different sizes of the layers. A smaller net with good driving behavior was choosen at the end.

The models were trained with images transformed to the colorspaces RGB and HSV. Models with using HSV colorspace were not better than using RGB so only RGB was used since the original output from the simulator is RGB and no transformation has to be done during runtime.

For regularization Dropout was used with a rate of 0.5 which worked well and was therefore not changed.

The optimizer used was Adam so no extra parameter tuning was done.

## Getting the training data

Training data was produced by driving the simulator manually with a controller. Using the mouse was too difficult and using only the keyboard was not accurat enough.

Both tracks of the simulator were used, in both two loops in both directions were driven. Additional corrections from wrong car trajectories were recorded and special situations like curves or difficult to detect situations on the tracks were used to have better data.


# Solution Design Approach

The first try neural network was based on a network architecture which was used by a team from NVidia to drive a car in real world. Since for this work the model has simpler conditions in the simulator the big fully connected layers were reduced in size and one layer was removed. Both, the original model and the reduced model were trained and both showed the same good driving behavior.

A second architecture tested was Inception inspired but not so deep and with only 3x3 and 5x5 convolutions.
Starting with only two convolutional layers with stride 2x2 the resulting network was very large because the size of the image was not enough reduced before the connection to the fully connected layers.
So finally five convolutional layers were used with an output of 32@(1x22).
Then the size of the fully connected layers was adapted to have a small net which still easy to train and which has a good driving behavior.

The second architecture was much smaller than the first architecture but has even a better driving behavior in difficult situations (strong curves with shadow).



## Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers:

![Network architecture][image_architecture]

## Creation of the Training Set & Training Process

To capture good driving behavior, first two laps of both tracks and in both direction were recorded. An example of track 1 is here:

![Manual driving][simulator_manual]

If the car is not in a good position recovery driving had to be trained. Therefore the car was driven manually in a wrong way and the recovery steering was recorded. For example driving too left from seen from the car's camera:

![Recovery][recovery]

Recording corrections is very time consuming so image augmentation was used for simple recovery situations like driving a bit too left or right. The car had not just a camera in the center but also one on the left and on the right. The images from these cameras are from a position in which the car would be too left or right. So the images were used with an adapted steering angle. The steering correction factor was set so that the car had a stable driving behavior.

As a simple addition flipped images with inverted steering angles were used:

![Flip][augmentation03]

Like in the NVidia paper additional images were created by virtually shifting the camera to the left and the right. The assumption was that all pixel before the horizon are on a plane and all pixels above the horizon have an infinite distance. So points on the horizon (and above) don't change the position and points directly before the camera shift like the camera. This was calculated as an affine transformation with OpenCV with two points at the horizon set fixed and one point at the bottom of the image shifted to the right or left.

For example an image of track one is show here:

![Flip][augmentation01]

The bottom left image is the original image form the center camera, the top right image is from the left camera. The top left image is virtually shifted at the same position as the left camera. It can be seen that the lane marking is the same in the realy shifted and virtually shifted camera images. Bottom right is a only half shifted image.

On the second track example the images at the top are from the center camera, the images at the bottom are from the left camera and in the middle the images are virtually shifted to the left (like in the left camera and only the half way).
It can be seen the the lane marking in the really and virtually shifted images are at the same positons. Distortions on the virtually shifted images can be seen at the objects at the right of the street. This happens since the assumptions that all pixels are on a plae doesn't hold.

![Flip][augmentation02]

After training the model and use it for autonomous driving in the simulator some difficult situations were found and extra training data was obtained driving manually with the simulator.

The following examples are recorded from the fron camera of the car. The rectangle is the image after cropping.

Here the model steered into the water:

![Difficult][difficult01]

Similar situation, the model had problems with the water on the left:

![Difficult][difficult02]

The first models steered straight forward to the sand track:

![Difficult][difficult03]

Here the model steered straight forward out of the track:

![Difficult][difficult04]

The same here:

![Difficult][difficult05]

The shadow disturbed the model so it started too late steering to the right:

![Difficult][difficult07]

After the final collection there were more than 60000 images recorded.

Image augmentation was only used for the center lane driving recordings, not for the difficult situations and recovery driving. In these cases the augmented images were not helpfull. For example if the car drives straight forward in a sharp left curve it should even turn faster if it drives not in the middle of the road but at the left side.


References:

[1] http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

[2] https://arxiv.org/abs/1409.4842
