import csv
import cv2
import numpy as np

lines = []
images = []
measurements = []

#with open('data/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)

#        image = cv2.imread(line[0])
#        images.append(image)

#        measurements.append(float(line[3]))

def process_image(img):
    return img

car_images, steering_angles = [], []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.3 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "data" # fill in the path to your training IMG directory
        img_center = process_image(np.asarray(cv2.imread(row[0])))
        img_left = process_image(np.asarray(cv2.imread(row[1])))
        img_right = process_image(np.asarray(cv2.imread(row[2])))

        # add images and angles to data set
        car_images.append(img_center)
        car_images.append(img_left)
        car_images.append(img_right)
        steering_angles.append(steering_center)
        steering_angles.append(steering_left)
        steering_angles.append(steering_right)

#print(measurements[0:10])

augmented_images, augmented_measurements = [], []

for image, measurement in zip(car_images, steering_angles):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, Lambda, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(filters=24, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=36, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=48, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='valid', activation='relu'))
model.add(Dropout(0.5))

#model.add(Conv2D(filters=20, kernel_size=[5,5] , strides=(1, 1), padding='valid', activation='relu'))
#model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print(y_train[0:10])

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model9.h5')