import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
from visiontransformer.model import VisionTransformer
from personal_lib import animator

def residual_block(X, f, filters, is_convolutional_block=False, s=1):
    X_shortcut = X

    # Als convolutional is, moet main path krimpen
    strides = (s, s) if is_convolutional_block else (1, 1)
    X = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=strides, padding='valid',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(f, f), strides=(1, 1), padding='same',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization()(X)

    # Als convolutional is, moet shortcut krimpen
    if is_convolutional_block:
        X_shortcut = tf.keras.layers.Conv2D(filters[2], (1, 1), strides=(s, s), padding='valid',
                            kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def make_model(r, depth):
    X_conv = tf.keras.layers.Input((2*r+1, 2*r+1, depth, 1))
    # smaller = r//20
    # smaller = 1
    # X = tf.keras.layers.AveragePooling3D(pool_size=(1, 1, 1))(X_conv)
    X = tf.keras.layers.Conv3D(16, (3, 3, 3), strides=(2, 2, 1), padding='same', kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X_conv)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.AveragePooling3D(pool_size=(1, 1, 1))(X)

    # X = residual_block(X, f=3, filters=[16, 16, 64], s=1, is_convolutional_block=True)
    # X = residual_block(X, 3, [16, 16, 64])
    # X = residual_block(X, f=3, filters=[16, 16, 64], s=2, is_convolutional_block=True)
    # X = residual_block(X, 3, [16, 16, 64])
    #
    # X = residual_block(X, f=3, filters=[16, 16, 64], s=2, is_convolutional_block=True)
    # X = residual_block(X, 3, [16, 16, 64])
    # X = residual_block(X, 3, [16, 16, 64])
    # X = residual_block(X, f=3, filters=[16, 16, 64], s=1, is_convolutional_block=True)
    # X = residual_block(X, 3, [16, 16, 64])

    # X = residual_block(X, f=3, filters=[16, 16, 64], s=2, is_convolutional_block=True)
    # X = residual_block(X, 3, [16, 16, 64])
    # X = residual_block(X, 3, [16, 16, 64])
    # X = residual_block(X, 3, [16, 16, 64])

    # X = residual_block(X, f=3, filters=[32, 32, 128], s=2, is_convolutional_block=True)
    # X = residual_block(X, 3, [32, 32, 128])
    # X = residual_block(X, 3, [32, 32, 128])
    # X = residual_block(X, 3, [32, 32, 128])
    #
    # X = residual_block(X, f=3, filters=[256, 256, 1024], s=1, is_convolutional_block=True)
    # X = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)
    X = tf.keras.layers.Dense(2048, "relu")(X)


    X = tf.keras.layers.Dense(depth)(X)

    model = tf.keras.models.Model(inputs=[X_conv], outputs=X, name='SortOfResNet50')
    return model

def make_model(r, depth):
    X_conv = tf.keras.layers.Input((2 * r + 1, 2 * r + 1, depth, 1))
    # X = tf.keras.layers.Conv3D(filters=16, kernel_size=(7, 7, 7), strides=(2,2,1), activation='relu', kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X_conv)
    X = residual_block(X_conv, f=3, filters=[16, 16, 16], is_convolutional_block=True, s=2)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)
    X = tf.keras.layers.Dense(1024, "relu")(X)

    X = tf.keras.layers.Dense(depth)(X)

    model = tf.keras.models.Model(inputs=[X_conv], outputs=X, name='SortOfResNet50')
    return model

def make_model(r, depth):
    model = VisionTransformer(
        image_size=2*r+1,
        patch_size=1,
        num_layers=32,
        num_classes=10,
        d_model=256,
        num_heads=16,
        mlp_dim=256,
        channels=depth,
        dropout=0.1,
    )
    return model

def make_model(r_a, r_b, r_c, r_d, depth):
    X_conv_a = tf.keras.layers.Input((2 * r_a + 1, 2 * r_a + 1, depth))

    filters = 128
    X = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))(X_conv_a)
    X = residual_block(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))(X)
    X_a = tf.keras.layers.Flatten()(X)


    X_conv_b = tf.keras.layers.Input((2 * r_b + 1, 2 * r_b + 1, depth))

    filters = 128
    X = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))(X_conv_b)
    X = residual_block(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters], s=2, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))(X)
    X_b = tf.keras.layers.Flatten()(X)


    X_conv_c = tf.keras.layers.Input((2 * r_c + 1, 2 * r_c + 1, depth))

    filters = 128
    X = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))(X_conv_c)
    X = residual_block(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters], s=1, is_convolutional_block=True)
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = residual_block(X, 3, [filters, filters, filters])
    X = tf.keras.layers.AveragePooling2D(pool_size=(1, 1))(X)
    X_c = tf.keras.layers.Flatten()(X)

    X_conv_d = tf.keras.layers.Input((2 * r_d + 1, 2 * r_d + 1, depth))
    X = tf.keras.layers.Flatten()(X_conv_d)

    d_neurons = 1024

    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X_d = tf.keras.layers.Dense(d_neurons, "relu")(X)

    concat = tf.keras.layers.Concatenate()([X_a, X_b, X_c, X_d]) #

    d_neurons = 1024

    X = tf.keras.layers.Dense(d_neurons, "relu")(concat)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)
    X = tf.keras.layers.Dense(d_neurons, "relu")(X)

    concat = tf.keras.layers.Lambda(lambda x: x/500)(concat) # get those 350_000 neurons to count for about 700

    X = tf.keras.layers.Concatenate()([concat, X])

    X = tf.keras.layers.Dense(depth)(X)

    model = tf.keras.models.Model(inputs=[X_conv_a, X_conv_b, X_conv_c, X_conv_d], outputs=X, name='SortOfResNet50')
    return model


