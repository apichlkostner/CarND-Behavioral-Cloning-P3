import csv
import cv2
import numpy as np
import pandas as pd

lines = []
images = []
measurements = []

def process_image(img):
    return img

#    for image, measurement in zip(car_images, steering_angles):
#        augmented_images.append(image)
#        augmented_measurements.append(measurement)
#        augmented_images.append(cv2.flip(image, 1))
#        augmented_measurements.append(measurement * -1.0)

def driveGenerator():
    df = pd.read_csv('data_06_corrections/driving_log.csv')
    batch_size = 32

    X_train = np.zeros([batch_size * 3, 160, 320, 3])
    y_train = np.zeros([batch_size * 3])

    nr_data = df.shape[0]

    print(nr_data)

    cnt = 0

    while 1:
        df.sample(frac=1).reset_index(drop=True)

        for i in range(int(nr_data / batch_size)):
            cnt += 1
            for j in range(batch_size):
                row = df.iloc[i * batch_size + j]
                
                steering_center = float(row[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.3
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                if cnt % 2 == 0:
                    img_center = process_image(np.asarray(cv2.imread(row[0])))
                    img_left = process_image(np.asarray(cv2.flip(cv2.imread(row[1]), 1)))
                    img_right = process_image(np.asarray(cv2.imread(row[2])))
                    X_train[j*3] = np.asarray(img_center)
                    y_train[j*3] = steering_center
                    X_train[j*3+1] = np.asarray(img_left)
                    y_train[j*3+1] = steering_left * -1.0
                    X_train[j*3+2] = np.asarray(img_right)
                    y_train[j*3+2] = steering_right
                else:
                    img_center = process_image(np.asarray(cv2.flip(cv2.imread(row[0]), 1)))
                    img_left = process_image(np.asarray(cv2.imread(row[1])))
                    img_right = process_image(np.asarray(cv2.flip(cv2.imread(row[2]), 1)))
                    X_train[j*3] = np.asarray(img_center)
                    y_train[j*3] = steering_center * -1.0
                    X_train[j*3+1] = np.asarray(img_left)
                    y_train[j*3+1] = steering_left
                    X_train[j*3+2] = np.asarray(img_right)
                    y_train[j*3+2] = steering_right * -1.0
            
            yield (X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, Lambda, MaxPooling2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(filters=24, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=36, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=48, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='valid', activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(driveGenerator(), steps_per_epoch = 340, epochs = 1, callbacks=[], validation_data=None)
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model12.h5')