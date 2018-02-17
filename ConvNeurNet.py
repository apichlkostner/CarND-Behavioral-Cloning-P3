def get_model_functional(input_shape, horizon, small_net = True):
    """
    Convolutional neural network using Keras functional API

    3 Layers with 3x3 and 5x5 convolution in parallel
    2 Layers with 3x3 convolution
    3 Dense layers

    Input: input_shape - shape of the input image
           horizon - horizon of images in pixels from top
           small_net - not used in this model
    """
    from keras.models import Model
    from keras.layers.merge import add, concatenate
    from keras.layers import Input, MaxPooling2D, Flatten, Dense, Conv2D, Dropout, Lambda, Cropping2D, BatchNormalization
    
    inputs = Input(shape=input_shape)

    # preprocessing: normalization and cropping
    preproc = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)
    preproc = Cropping2D(cropping=((horizon, 16), (0, 0)))(preproc)

    # layer 1: convolution 3x3 and 5x5
    x = Conv2D(filters=12, kernel_size=[5,5] , strides=(2, 2), padding='same', activation='elu')(preproc)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    layer1_1 = x
    x = Conv2D(filters=12, kernel_size=[3,3] , strides=(2, 2), padding='same', activation='elu')(preproc)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    layer1_2 = x
    layer1 = concatenate([layer1_1, layer1_2])

    # layer 2: convolution 3x3 and 5x5
    x = Conv2D(filters=18, kernel_size=[5,5] , strides=(2, 2), padding='same', activation='elu')(layer1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    layer2_1 = x
    x = Conv2D(filters=18, kernel_size=[3,3] , strides=(2, 2), padding='same', activation='elu')(layer1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    layer2_2 = x
    layer2 = concatenate([layer2_1, layer2_2])

    # layer 3: convolution 3x3 and 5x5
    x = Conv2D(filters=24, kernel_size=[5,5] , strides=(2, 2), padding='same', activation='elu')(layer2)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    layer3_1 = x
    x = Conv2D(filters=24, kernel_size=[3,3] , strides=(2, 2), padding='same', activation='elu')(layer2)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    layer3_2 = x
    layer3 = concatenate([layer3_1, layer3_2])

    # layer 4: convolution 3x3
    x = Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='valid', activation='elu')(layer3)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # layer 5: convolution 3x3
    x = Conv2D(filters=32, kernel_size=[3,3] , strides=(1, 1), padding='valid', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Flatten
    x = Flatten()(x)

    # 3 dense layers with elu activation
    x = Dense(200, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(50, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(30, activation='elu')(x)
    x = BatchNormalization()(x)

    # output with linear activation
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse', optimizer='adam')

    return model

def get_model_sequential(input_shape, horizon, small_net = True):
    """
    Creates a model

    Input: input_shape - shape of the input image
           horizon - horizon of images in pixels from top
           small_net - if true only small fully connected layers are used
    """
    from keras.models import Sequential
    from keras.layers import MaxPooling2D, Flatten, Dense, Conv2D, Dropout, Lambda, Cropping2D, BatchNormalization
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = input_shape))
    model.add(Cropping2D(cropping=((horizon, 16), (0, 0))))
    model.add(Conv2D(filters=24, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=36, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=48, kernel_size=[5,5] , strides=(2, 2), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))      
    model.add(Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64, kernel_size=[3,3] , strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    if small_net:
        # reduced fully connected layers
        model.add(Flatten())
        model.add(Dense(200, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(30, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
    else:
        # like in NVidia paper
        model.add(Flatten())
        model.add(Dense(1164, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model