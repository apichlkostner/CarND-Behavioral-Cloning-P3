import csv
import cv2
import numpy as np
import pandas as pd

def process_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class DriveImageGenerator:
    augment_mult = 2

    def __init__(self):
        self.df_drive = {'train': None, 'valid': None}
        self.len = {'train': 0, 'valid': 0}
        self.val_split = 0.0
        self.batch_size = 0

    def fit(self, df_filenames=['~/data_track_1/driving_log.csv'], batch_size=32, val_split=0.2):
        df = pd.DataFrame()
        for df_filename in df_filenames:
            df = df.append(pd.read_csv(df_filename, header=None))

        df.columns = ['center', 'left', 'right', 'steering', 'a', 'b', 'c']

        # split data in center, left and right images
        dfc = df[['center', 'steering']].copy()
        dfl = df[['left', 'steering']].copy()
        dfr = df[['right', 'steering']].copy()

        # create adjusted steering measurements for the side camera images
        correction = 0.19
        dfl['steering'] += correction
        dfr['steering'] -= correction

        dfc.columns = ['image', 'steering']
        dfl.columns = ['image', 'steering']
        dfr.columns = ['image', 'steering']

        # append all images to one big data frame
        dfn = dfc.append(dfl).append(dfr)

        print("Dataframe size = " + str(dfn.shape))
        
        # random shuffle
        dfn.sample(frac=1).reset_index(drop=True)

        # train / validation split
        pos_split = int(val_split * dfn.shape[0])

        self.df_drive['train'] = dfn[pos_split:]
        self.df_drive['valid'] = dfn[0:pos_split]
        
        self.len = {'train': self.df_drive['train'].shape[0], 'valid': self.df_drive['valid'].shape[0]}

    def flow(self, data_set_name):
        X_data = np.zeros([batch_size, 160, 320, 3])
        y_data = np.zeros([batch_size])

        self.batch_size = batch_size

        while 1:
            # random shuffle of train set
            df_drive = self.df_drive[data_set_name].sample(frac=1).reset_index(drop=True)

            for i in range(int(self.len[data_set_name] / batch_size)):
                for j in range(batch_size):
                    row = df_drive.iloc[i * batch_size + j]
                    
                    steering = float(row['steering'])

                    #print(steering)

                    aug_technique = np.random.randint(0, 2)

                    # simple data augmentation: flip images and steering angle
                    if aug_technique == 0:
                        img = process_image(np.asarray(cv2.imread(row['image'])))
                        X_data[j] = np.asarray(img)
                        y_data[j] = steering
                    else:
                        img = process_image(np.asarray(cv2.flip(cv2.imread(row['image']), 1)))
                        X_data[j] = np.asarray(img)
                        y_data[j] = steering * -1.0
                
                yield (X_data, y_data)

    def num_samples(self, data_set_name):
        return self.len[data_set_name] * self.augment_mult


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
#dg.fit(df_filename = 'data_06_corrections/driving_log.csv', batch_size = batch_size)
df_filenames = ['~/data_track_1/driving_log.csv', '~/data_track_2/driving_log.csv']
dg.fit(df_filenames=df_filenames, batch_size = batch_size)

steps_train = int((dg.num_samples('train')) / batch_size) + 1
steps_valid = int((dg.num_samples('valid')) / batch_size) + 1

model.fit_generator(dg.flow('train'), steps_per_epoch = steps_train, 
                    validation_data=dg.flow('valid'), validation_steps=steps_valid,
                    epochs = 3, callbacks=[])

model.save('model_track_1_and_2_01_driving.h5')