import csv
import cv2
import numpy as np
import pandas as pd
from ImageGenerator import DriveImageGenerator
from ConvNeurNet import get_model_functional, get_model_sequential

# configuration
HORIZON = 40    # horizon from top in pixels
LOAD_MODEL = False
LOAD_MODEL_NAME = 'name.hdf5'
BATCH_SIZE = 64
EPOCHS     = 120
SMALL_NET = False   # Use smaller fully connected layers
GET_MODEL = get_model_functional
AUGMENTATION = True
TARGET_INPUT_SIZE = (100, 200, 3)
COLORSPACE = cv2.COLOR_BGR2RGB      # function for colorspace conversion
# reversed for cv2
TARGET_IMAGE_SIZE = (TARGET_INPUT_SIZE[1], TARGET_INPUT_SIZE[0])

# log files and steering correction used for side cameras
LOGS = [{'filename': '~/data/track_1/optimal_01/driving_log.csv', 'steeringCorrection': 0.1},
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

def main():
    from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
    from keras.models import load_model
    
    if LOAD_MODEL:
        import keras.backend as K

        # load existing model and start with reduced learning rate
        model = load_model(LOAD_MODEL_NAME)
        K.set_value(model.optimizer.lr, 1e-04)
    else:
        # create model
        model = GET_MODEL(input_shape=TARGET_INPUT_SIZE, horizon=HORIZON, small_net=SMALL_NET)

    # reads images from disc and does on the fly augmentation
    dg = DriveImageGenerator()

    # preprocessing of log files
    dg.fit(logs=LOGS, batch_size=BATCH_SIZE, target_image_size=TARGET_IMAGE_SIZE,
            augmentation=AUGMENTATION, colorspace=COLORSPACE)

    # number of steps for training and validation
    steps_train = int((dg.get_num_samples('train')) / BATCH_SIZE) + 1
    steps_valid = int((dg.get_num_samples('valid')) / BATCH_SIZE) + 1

    name = 'model_func_large_01'

    # log history
    historyLogger = CSVLogger('history/' + name + '.csv', append=True)

    # safe model checkpoints
    modelCheckPoint = ModelCheckpoint('checkpoints/' + name + '_{epoch:02d}-{val_loss:.4f}.hdf5',
                                        monitor='val_loss', verbose=0, save_best_only=True)

    # stop training when validation loss is not decreasing
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20)

    # list with callbacks for fit_generator
    callbacks = [historyLogger, modelCheckPoint, earlyStopping]

    # fit model
    model.fit_generator(dg.flow('train'), steps_per_epoch = steps_train, 
                        validation_data=dg.flow('valid'), validation_steps=steps_valid,
                        epochs = EPOCHS, callbacks=callbacks)

if __name__ == "__main__":
    main()