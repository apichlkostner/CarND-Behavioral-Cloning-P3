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

class DriveImageGenerator:
    augment_mult = 2

    def __init__(self):
        self.df_train = None
        self.df_valid = None
        self.len_train = 0
        self.len_valid = 0
        self.val_split = 0.0
        self.batch_size = 0

    def fit(self, df_filename='data/driving_log.csv', batch_size=32, val_split=0.2):
        df = pd.read_csv(df_filename, header=None)
        df.columns = ['center', 'left', 'right', 'steering', 'a', 'b', 'c']

        # split data in center, left and right images
        dfc = df[['center', 'steering']].copy()
        dfl = df[['left', 'steering']].copy()
        dfr = df[['right', 'steering']].copy()

        # create adjusted steering measurements for the side camera images
        correction = 0.3
        dfl['steering'] += correction
        dfr['steering'] -= correction

        dfc.columns = ['image', 'steering']
        dfl.columns = ['image', 'steering']
        dfr.columns = ['image', 'steering']

        # append all images to one big data frame
        dfn = dfc.append(dfl).append(dfr)

        print(dfc.iloc[0]['image'])
        print(dfl.iloc[0]['image'])
        print(dfr.iloc[0]['image'])
        print("Dataframe size = " + str(dfn.shape))
        
        # random shuffle
        dfn.sample(frac=1).reset_index(drop=True)

        # train / validation split
        pos_split = int(val_split * dfn.shape[0])

        self.df_train = dfn[pos_split:]
        self.df_valid = dfn[0:pos_split]
        
        self.len_train = self.df_train.shape[0]
        self.len_valid = self.df_valid.shape[0]


    def flow_train(self):
        X_train = np.zeros([batch_size, 160, 320, 3])
        y_train = np.zeros([batch_size])

        cnt = 0
        self.batch_size = batch_size

        while 1:
            # random shuffle of train set
            self.df_train.sample(frac=1).reset_index(drop=True)

            for i in range(int(self.len_train / batch_size)):
                cnt += 1
                for j in range(batch_size):
                    row = self.df_train.iloc[i * batch_size + j]
                    
                    steering = float(row['steering'])

                    # simple data augmentation: flip images and steering angle
                    if cnt % 2 == 0:
                        img = process_image(np.asarray(cv2.imread(row['image'])))
                        X_train[j] = np.asarray(img)
                        y_train[j] = steering
                    else:
                        img = process_image(np.asarray(cv2.flip(cv2.imread(row['image']), 1)))
                        X_train[j] = np.asarray(img)
                        y_train[j] = steering * -1.0
                
                yield (X_train, y_train)

    def flow_valid(self):
        X_train = np.zeros([batch_size, 160, 320, 3])
        y_train = np.zeros([batch_size])

        cnt = 0

        while 1:
            # random shuffle of train set
            self.df_valid.sample(frac=1).reset_index(drop=True)

            for i in range(int(self.len_valid / batch_size)):
                cnt += 1
                for j in range(batch_size):
                    row = self.df_valid.iloc[i * batch_size + j]
                    
                    steering = float(row['steering'])

                    if cnt % 2 == 0:
                        img = process_image(np.asarray(cv2.imread(row['image'])))
                        X_train[j] = np.asarray(img)
                        y_train[j] = steering
                    else:
                        img = process_image(np.asarray(cv2.flip(cv2.imread(row['image']), 1)))
                        X_train[j] = np.asarray(img)
                        y_train[j] = steering * -1.0
                
                yield (X_train, y_train)

    def num_samples(self):
        return [self.len_train * self.augment_mult, self.len_valid * self.augment_mult]


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

batch_size = 64

dg = DriveImageGenerator()
dg.fit(df_filename = 'data_06_corrections/driving_log.csv', batch_size = batch_size)
print("Anzahl smaples" + str(dg.num_samples()))
steps_train = int((dg.num_samples())[0] / batch_size) + 1
steps_valid = int((dg.num_samples())[1] / batch_size) + 1

model.fit_generator(dg.flow_train(), steps_per_epoch = steps_train, 
                    validation_data=dg.flow_valid(), validation_steps=steps_valid,
                    epochs = 1, callbacks=[])

model.save('model12.h5')