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
        self.num_samples = {'train': 0, 'valid': 0}
        self.val_split = 0.0
        self.batch_size = 0

    def fit(self, df_filenames=['~/data_track_1/driving_log.csv'], batch_size=32, val_split=0.2):
        df = pd.DataFrame()
        for df_filename in df_filenames:
            # read first log
            temp_df = pd.read_csv(df_filename, header=None)

            # add column names
            temp_df.columns = ['center', 'left', 'right', 'steering', 'a', 'b', 'c']

            # split data in center, left and right images
            temp_df_center = temp_df[['center', 'steering']].copy()
            temp_df_left = temp_df[['left', 'steering']].copy()
            temp_df_right =temp_df[['right', 'steering']].copy()

            # create adjusted steering measurements for the side camera images
            # track 2 needs larger value than track 1
            if df_filename.find('track_1') != -1:
                correction = 0.19
            else:
                correction = 0.4

            print('Correction = ' + str(correction))

            temp_df_left['steering'] += correction
            temp_df_right['steering'] -= correction

            temp_df_center.columns = ['image', 'steering']
            temp_df_left.columns = ['image', 'steering']
            temp_df_right.columns = ['image', 'steering']

            # append all images to one big data frame
            df = df.append(temp_df_center).append(temp_df_left).append(temp_df_right)

        print("Dataframe size = " + str(df.shape))
        
        # random shuffle
        df = df.sample(frac=1).reset_index(drop=True)

        # train / validation split
        pos_split = int(val_split * df.shape[0])

        self.df_drive['train'] = df[pos_split:]
        self.df_drive['valid'] = df[0:pos_split]
        
        self.num_samples = {'train': self.df_drive['train'].shape[0], 'valid': self.df_drive['valid'].shape[0]}
        print('Number of samples: ' + str(self.num_samples))

    def flow(self, data_set_name):
        X_data = np.zeros([batch_size, 160, 320, 3])
        y_data = np.zeros([batch_size])

        self.batch_size = batch_size

        while 1:
            # random shuffle of train set
            df_drive = self.df_drive[data_set_name].sample(frac=1).reset_index(drop=True)

            for i in range(int(self.num_samples[data_set_name] / batch_size)):
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

    def get_num_samples(self, data_set_name):
        return self.num_samples[data_set_name] * self.augment_mult


from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, Lambda, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

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

# log files for track 1 and track 2
df_filenames = ['~/data_track_1/driving_log.csv', '~/data_track_1_2/driving_log.csv', '~/data_track_2_2/driving_log.csv']

# preprocessing of log files
dg.fit(df_filenames=df_filenames, batch_size = batch_size)

steps_train = int((dg.get_num_samples('train')) / batch_size) + 1
steps_valid = int((dg.get_num_samples('valid')) / batch_size) + 1

# safe model checkpoints
modelCheckPoint = ModelCheckpoint('model_10_{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True)
# stop training when validation loss is not decreasing
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2)

callbacks = [modelCheckPoint, earlyStopping]

# fit model
model.fit_generator(dg.flow('train'), steps_per_epoch = steps_train, 
                    validation_data=dg.flow('valid'), validation_steps=steps_valid,
                    epochs = 5, callbacks=callbacks)
