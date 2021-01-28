import tensorflow as tf


def vgg_block(in_tensor, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(in_tensor)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    o = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return o

def inception_block(input_tensor, num_filters):
    p1 = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    p1 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p1)
    
    p2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
    p2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p2)
    
    p3 = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    
    # return tf.keras.layers.concatenate([p1, p2, p3], axis=3)
    o = tf.keras.layers.Add()([p1, p2, p3])
    
    return o
    

def get_model(depth, learning_rate, base, patch_size, num_classes=1, backend='vgg'):

    input_ = tf.keras.layers.Input(shape=patch_size)
    x = tf.keras.layers.Conv2D(base, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_)
    # x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    fm = base
    for b in range(depth-1):
        fm = fm * 2
        if backend == 'vgg':
            x = vgg_block(x, fm)
        elif backend == 'inception':
            x = inception_block(x, fm)
        else:
            raise Exception('unkonwn backend type. allowed vales are vgg/inception')

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(200, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    Out = tf.keras.layers.Dense(num_classes, activation='softmax', name='Out')(x)

    model = tf.keras.models.Model(inputs=[input_], outputs=[Out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])

    return model


def vgg_keras_application(fine_tune, num_classes, learning_rate, patch_size):
    
    # this could also be the output a different Keras model or layer
    input_tensor = tf.keras.layers.Input(shape=patch_size)
    # create the base pre-trained model
    base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                                   include_top=False,
                                                   input_tensor=input_tensor)
    
    if fine_tune is True:
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
    # add a global spatial average pooling layer
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = tf.keras.layers.Dense(200, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # this is the model we will train
    model = tf.keras.models.Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                          loss=['categorical_crossentropy'],
                          metrics=['accuracy'])
    
    return model


def inception_v3_keras_application(fine_tune, num_classes, learning_rate, patch_size):
    # this could also be the output a different Keras model or layer
    input_tensor = tf.keras.layers.Input(shape=patch_size)
    # create the base pre-trained model
    base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet',
                                                                include_top=False,
                                                                input_tensor=input_tensor)
    
    if fine_tune is True:
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
    # add a global spatial average pooling layer
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = tf.keras.layers.Dense(200, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # this is the model we will train
    model = tf.keras.models.Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                          loss=['categorical_crossentropy'],
                          metrics=['accuracy'])
    
    return model


def Xception_keras_application(fine_tune, num_classes, learning_rate, patch_size,):
    # this could also be the output a different Keras model or layer
    input_tensor = tf.keras.layers.Input(shape=patch_size)
    # create the base pre-trained model
    base_model = tf.keras.applications.xception.Xception(weights='imagenet',
                                                                include_top=False,
                                                                input_tensor=input_tensor)
    
    if fine_tune is True:
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
    # add a global spatial average pooling layer
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = tf.keras.layers.Dense(200, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # this is the model we will train
    model = tf.keras.models.Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                          loss=['categorical_crossentropy'],
                          metrics=['accuracy'])
    
    return model


def get_imagenet_model(learning_rate, patch_size, num_classes=2, model_type='vgg16', fine_tune=True):
    if model_type == 'vgg':
        return vgg_keras_application(fine_tune, num_classes, learning_rate, patch_size)
    elif model_type == 'inception':
        return inception_v3_keras_application(fine_tune, num_classes, learning_rate, patch_size)

    elif model_type == 'xception':
        return Xception_keras_application(fine_tune, num_classes, learning_rate, patch_size)
    else:
        print(model_type)
        raise Exception('unknown model_type')
        

if __name__ == '__main__':
    M = get_model(depth=6, learning_rate=1e-4, base=32, patch_size=(224, 224, 3), num_classes=2)
    M.summary()
