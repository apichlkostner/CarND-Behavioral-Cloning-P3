import csv
import cv2
import numpy as np
import pandas as pd



class DriveImageGenerator:
    """ Image generator for Keras """

    def __init__(self):
        self.df_drive = {'train': None, 'valid': None}
        self.num_samples = {'train': 0, 'valid': 0}
        self.val_split = 0.0
        self.batch_size = 0
        self.horizon = 0
        self.target_image_size = (0, 0)
        self.augmentation = False

    def process_image(self, img):
        """ preprocessing of the image """
        img = cv2.resize(img, self.target_image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def shift_camera(self, img, horizon, shift):
        """
        Transforms the image as if the camera was shifted
        
        input: img - image to process
               horizon - horizon in the image in pixels from top
               shift - shift of the camera in pixels, positive to the right
        """
        rows, cols, _ = img.shape

        # horizon is fix
        pts1 = np.float32([[0, 0], [cols, 0], [int(cols / 2),         rows - horizon]])
        # shift to the right -> points shift to the left
        pts2 = np.float32([[0, 0], [cols, 0], [int(cols / 2) - shift, rows - horizon]])

        M = cv2.getAffineTransform(pts1, pts2)

        img[horizon:, :] = cv2.warpAffine(img[horizon:, :], M, (cols, rows - horizon))

        return img

    def fit(self, logs, batch_size=32, val_split=0.2, horizon=40, target_image_size=(100, 200), augmentation=True):
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
        self.target_image_size = target_image_size
        self.augmentation = augmentation

        df = pd.DataFrame()
        log_nr_counter = 0 # counter for saving sample images

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
            if log_nr_counter == 3:                
                img_center = cv2.imread((temp_df_center.iloc[524])['image']).copy()
                img_left = cv2.imread((temp_df_left.iloc[524])['image']).copy()
                img_shift = self.shift_camera(img_center.copy(), 70, -60)
                img_shift_middle = self.shift_camera(img_center.copy(), 70, -30)
                cv2.imwrite('img_center03.png', img_center)
                cv2.imwrite('img_left03.png', img_left)
                cv2.imwrite('img_shift03.png', img_shift)
                cv2.imwrite('img_shift_middle03.png', img_shift_middle)

            log_nr_counter += 1
            
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
                    img = self.process_image(cv2.imread(row['image']))

                    if self.augmentation:
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