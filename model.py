import csv
import cv2
import numpy as np
import pandas as pd

# configuration
HORIZON = 40    # horizon from top in pixels
LOAD_MODEL = False
BATCH_SIZE = 64
EPOCHS     = 120
SMALL_NET = True   # Use smaller fully connected layers
TARGET_INPUT_SIZE = (100, 200, 3)
# reversed for cv2
TARGET_IMAGE_SIZE = (TARGET_INPUT_SIZE[1], TARGET_INPUT_SIZE[0])

def process_image(img):
    """ preprocessing of the image """
    img = cv2.resize(img, TARGET_IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class DriveImageGenerator:
    """ Image generator for Keras """

    def __init__(self):
        self.df_drive = {'train': None, 'valid': None}
        self.num_samples = {'train': 0, 'valid': 0}
        self.val_split = 0.0
        self.batch_size = 0
        self.horizon = 0

    def shift_camera(self, img, horizon, shift):
        """
        Transforms the image as if the camera was shifted
        
        input: img - image to process
               horizon - horizon in the image in pixels from top
               shift - shift of the camera in pixels, positive to the right
        """
        rows, cols, ch = img.shape

        # horizon is fix
        pts1 = np.float32([[0, 0], [cols, 0], [int(cols / 2),         rows - horizon]])
        # shift to the right -> points shift to the left
        pts2 = np.float32([[0, 0], [cols, 0], [int(cols / 2) - shift, rows - horizon]])

        M = cv2.getAffineTransform(pts1, pts2)

        img[horizon:, :] = cv2.warpAffine(img[horizon:, :], M, (cols, rows - horizon))

        return img

    def fit(self, logs, batch_size=32, val_split=0.2, horizon=40):
        """
        Fits the generator to the data sets

        input: logs - list of maps with logfiles
                             'filename': filename of the csv
                             'steeringCorrection': steering correction for left and right
                                                   camera images. If 0 no augmentation
                                                   should be done
                batch_size - batch size
                val_split - proportion of validation data (for split)
                horizon - horizon of image in pixels from top
        """
        self.batch_size = batch_size
        self.horizon = horizon

        df = pd.DataFrame()

        for logs in logs:
            # read first log
            temp_df = pd.read_csv(logs['filename'], header=None)

            # add column names
            temp_df.columns = ['center', 'left', 'right', 'steering', 'a', 'b', 'c']

            # is steeringCorrection is zero the side cameras shouldn't be used
            useSideCameras = logs['steeringCorrection'] > 0.0

            # if side cameras shouldn't be used augmentation is also forbidden
            temp_df['augmentation'] = useSideCameras

            # split data in center, left and right images
            info_columns = ['steering', 'augmentation']
            temp_df_center = temp_df[['center'] + info_columns].copy()
            temp_df_left = temp_df[['left'] + info_columns].copy()
            temp_df_right =temp_df[['right'] + info_columns].copy()

            # create adjusted steering measurements for the side camera images            
            print('Reading logfile ' + logs['filename'])
            print('Correction factor = ' + str(logs['steeringCorrection']))

            temp_df_left['steering'] += logs['steeringCorrection']
            temp_df_right['steering'] -= logs['steeringCorrection']

            # change names of columns
            cols_new = ['image', 'steering', 'augmentation']

            temp_df_center.columns = cols_new
            temp_df_left.columns = cols_new
            temp_df_right.columns = cols_new

            # save samples from augmentation (debug)
            if False:
                img_center = cv2.imread((temp_df_center.iloc[1])['image']).copy()
                img_left = cv2.imread((temp_df_left.iloc[1])['image']).copy()
                img_shift = self.shift_camera(img_center.copy(), 70, -60)
                img_shift_middle = self.shift_camera(img_center.copy(), 70, -30)
                cv2.imwrite('img_center.png', img_center)
                cv2.imwrite('img_left.png', img_left)
                cv2.imwrite('img_shift.png', img_shift)
                cv2.imwrite('img_shift_middle.png', img_shift_middle)

            # append all images to one big data frame
            if useSideCameras:
                df = df.append(temp_df_center).append(temp_df_left).append(temp_df_right)
            else:
                df = df.append(temp_df_center)

        # random shuffle
        df = df.sample(frac=1).reset_index(drop=True)

        # train / validation split
        pos_split = int(val_split * df.shape[0])

        self.df_drive['train'] = df[pos_split:]
        self.df_drive['valid'] = df[0:pos_split]
        
        self.num_samples = {'train': self.df_drive['train'].shape[0], 'valid': self.df_drive['valid'].shape[0]}
        
        print('Number of samples: ' + str(self.num_samples))

    def flow(self, data_set_name):
        """
        Generates batches in an endless loop

        Input: data_set_name - name of dataset ('train' or 'valid')
        """
        X_data = np.zeros([self.batch_size, 100, 200, 3])
        y_data = np.zeros([self.batch_size])
        
        while 1:
            # random shuffle of train set
            df_drive = self.df_drive[data_set_name].sample(frac=1).reset_index(drop=True)

            # loop over number of batches
            for batch_nr in range(int(self.num_samples[data_set_name] / self.batch_size)):
                # loop over samples in batch
                for sample in range(self.batch_size):
                    row = df_drive.iloc[batch_nr * self.batch_size + sample]
                    
                    steering = float(row['steering'])

                    # preprocessing of image
                    img = process_image(cv2.imread(row['image']))

                    # simple data augmentation: flip images and steering angle
                    flip = (np.random.randint(0, 2) == 1)

                    if flip:
                        img = cv2.flip(img, 1)
                        steering *= -1.0
                    
                    # augmentation: shift camera
                    if row['augmentation']:
                        # 0: no shift, 1: left shift, 2: right shift
                        shift = np.random.randint(0, 3)

                        if shift == 1:
                            img = self.shift_camera(img, self.horizon, -20)
                            steering += 0.03
                        
                        if shift == 2:                
                            img = self.shift_camera(img, self.horizon, 20)
                            steering -= 0.03

                    X_data[sample] = img
                    y_data[sample] = steering
                
                yield (X_data, y_data)

    def get_num_samples(self, data_set_name):
        """
        Get the number of samples

        Input: data_set_name - name of dataset ('train' or 'valid')
        """
        return self.num_samples[data_set_name]


def get_model(input_shape, horizon, small_net = True):
    """
    Creates a model

    Input: input_shape - shape of the input image
           horizon - horizon of images in pixels from top
           small_net - if true only small fully connected layers are used
    """
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Conv2D, Dropout, Lambda, Cropping2D, BatchNormalization
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = input_shape))
    model.add(Cropping2D(cropping=((horizon, 16), (0, 0))))
    model.add(Conv2D(filters=24, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=36, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=48, kernel_size=[5,5] , strides=(2, 2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    if small_net:
        # reduced fully connected layers
        model.add(Flatten())
        model.add(Dense(200))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(30))
        model.add(Dense(1))
    else:
        # like in NVidia paper
        model.add(Flatten())
        model.add(Dense(1164))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def main():
    from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
    from keras.models import load_model
    
    if LOAD_MODEL:
        import keras.backend as K

        # load existing model and start with reduced learning rate
        model = load_model('name.hdf5')
        K.set_value(model.optimizer.lr, 1e-04)
    else:
        # create model
        model = get_model(input_shape=TARGET_INPUT_SIZE, horizon=HORIZON, small_net=SMALL_NET)

    # reads images from disc and does on the fly augmentation
    dg = DriveImageGenerator()

    # log files
    logs = [{'filename': '~/data/track_1/optimal_01/driving_log.csv', 'steeringCorrection': 0.1},
            {'filename': '~/data/track_1/critical_situations/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_1/corrections_01/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_2/optimal_middle_01/driving_log.csv', 'steeringCorrection': 0.1},
            {'filename': '~/data/track_2/critical_situations_middle_01/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_2/critical_situations_middle_02/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_2/corrections_middle_01/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_2/corrections_middle_02/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_2/corrections_middle_03/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_2/corrections_middle_03/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_2/corrections_middle_04/driving_log.csv', 'steeringCorrection': 0.0},
            {'filename': '~/data/track_2/corrections_middle_04/driving_log.csv', 'steeringCorrection': 0.0}
            ]


    # preprocessing of log files
    dg.fit(logs=logs, batch_size=BATCH_SIZE)

    # number of steps for training and validation
    steps_train = int((dg.get_num_samples('train')) / BATCH_SIZE) + 1
    steps_valid = int((dg.get_num_samples('valid')) / BATCH_SIZE) + 1

    name = 'model_small_04_horizon40'

    # log history
    historyLogger = CSVLogger('history/' + name + '.csv', append=True)

    # safe model checkpoints
    modelCheckPoint = ModelCheckpoint('checkpoints/' + name + '_{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True)

    # stop training when validation loss is not decreasing
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30)

    # list with callbacks for fit_generator
    callbacks = [historyLogger, modelCheckPoint, earlyStopping]

    # fit model
    model.fit_generator(dg.flow('train'), steps_per_epoch = steps_train, 
                        validation_data=dg.flow('valid'), validation_steps=steps_valid,
                        epochs = EPOCHS, callbacks=callbacks)

if __name__ == "__main__":
    main()