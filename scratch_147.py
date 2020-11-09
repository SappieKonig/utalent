import tensorflow as tf

def residual_block(X, f, filters, is_convolutional_block=False, s=1):
    X_shortcut = X

    # Als convolutional is, moet main path krimpen
    strides = (s, s, s) if is_convolutional_block else (1, 1, 1)
    X = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1, 1, 1), strides=strides, padding='valid',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv3D(filters=filters[1], kernel_size=(f, f, f), strides=(1, 1, 1), padding='same',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv3D(filters=filters[2], kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    # Als convolutional is, moet shortcut krimpen
    if is_convolutional_block:
        X_shortcut = tf.keras.layers.Conv3D(filters[2], (1, 1, 1), strides=(s, s, s), padding='valid',
                            kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def make_model(r, depth):
    X_conv = tf.keras.layers.Input((2*r+1, 2*r+1, depth, 1))

    X = tf.keras.layers.ZeroPadding3D((3, 3, 3))(X_conv)

    X = tf.keras.layers.Conv3D(256, (7, 7, 7), strides=(2, 2, 2), padding='same', kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    # X = tf.keras.layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(X)

    X = residual_block(X, f=3, filters=[64, 64, 256], s=1, is_convolutional_block=True)

    X = residual_block(X, f=3, filters=[64, 64, 256], s=2, is_convolutional_block=True)
    X = residual_block(X, 3, [64, 64, 256])

    X = residual_block(X, f=3, filters=[256, 256, 1024], s=1, is_convolutional_block=True)
    X = residual_block(X, 3, [256, 256, 1024])

    X = residual_block(X, f=3, filters=[512, 512, 2048], s=1, is_convolutional_block=True)
    X = tf.keras.layers.AveragePooling3D()(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(512, activation="selu")(X)

    X = tf.keras.layers.Dense(depth, activation='selu', kernel_initializer=tf.keras.initializers.he_uniform(seed=0))(X)

    model = tf.keras.models.Model(inputs=[X_conv], outputs=X, name='SortOfResNet50')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss='mse')
    model.summary()
    return model


import numpy as np
model = tf.keras.models.load_model("C:/tensorTestModels/baaaad_convergence_mapper")


mass = np.load("C:/datasets/4096x4096x10_mass.npy")
conv = np.load("C:/datasets/4096x4096x10_convergence.npy")

r = 10
depth = 10
dia = 2*r+1 # diameter
ran = np.tile(np.arange(dia), dia)
var1 = ran.reshape((dia, dia), order="F").flatten()
var2 = ran.reshape((dia, dia), order="C").flatten()
options1 = np.array([var1 + i for i in range(4096-dia)])
options2 = np.array([var2 + i for i in range(4096-dia)])

model = make_model(r, depth)

for i in range(1000):
    model.save("C:/tensorTestModels/baaaad_convergence_mapper")

    choices = np.random.randint(4096-dia, size=(2,10000))

    net_input = mass[options1[choices[0]], options2[choices[1]]].reshape(-1, dia, dia, depth, 1)
    net_output = conv[choices[0]+r, choices[1]+r].reshape(-1, depth)
    print(net_input.shape, net_output.shape)

    model.fit(net_input, net_output, batch_size=32, validation_split=.2)