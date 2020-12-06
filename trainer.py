import numpy as np
import animator
from model import make_model
import tqdm
import animator

# the following loads a model from memory to train on
model = tf.keras.models.load_model("/home/ignace/tensorTestModels/please_just_work_for_once_2D_v2")

# we have 4 different radii for different levels of precision
r_a = 40
r_b = 20
r_c = 10
r_d = 5

# the np.load() function loads the dataset into memory for further use
mass = np.load("/home/ignace/datasets/16384x16384x10_real_lmstellar.npy")
conv = np.load("/home/ignace/datasets/16384x16384x10_real_kappa.npy")
indices = np.load("/home/ignace/datasets/indices.npy")

# since we need to be able to pick a pixel that has enough room around it for context,
# we can not pick from the edges of our map, since there will not be enough room left
# this code trims the possible galaxies we can choose from to only include those with extra headroom for context
valid_indices = (indices[:, 0] >= r_a) * (indices[:, 0] < len(mass)-r_a) * (indices[:, 1] >= r_a) * (indices[:, 1] < len(mass)-r_a)
indices = indices[valid_indices]

depth = 10
dia_a = 2*r_a+1
dia_b = 2*r_b+1
dia_c = 2*r_c+1
dia_d = 2*r_d+1

# these long lists of code return the indices of the context for the four different radii
# so let's say the context diameter is 3, and our pixel is at position (1, 1)
# then as context, we also want the pixels at positions: 
# (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
# this piece of code selects those pixels

ran = np.tile(np.arange(dia_a), dia_a)
var1_a = ran.reshape((dia_a, dia_a), order="F").flatten()
var2_a = ran.reshape((dia_a, dia_a), order="C").flatten()
options1_a = np.array([var1_a + i + r_a - r_a for i in range(len(mass)-2*r_a)])
options2_a = np.array([var2_a + i + r_a - r_a for i in range(len(mass)-2*r_a)])

ran = np.tile(np.arange(dia_b), dia_b)
var1_b = ran.reshape((dia_b, dia_b), order="F").flatten()
var2_b = ran.reshape((dia_b, dia_b), order="C").flatten()
options1_b = np.array([var1_b + i + r_a - r_b for i in range(len(mass)-2*r_a)])
options2_b = np.array([var2_b + i + r_a - r_b for i in range(len(mass)-2*r_a)])

ran = np.tile(np.arange(dia_c), dia_c)
var1_c = ran.reshape((dia_c, dia_c), order="F").flatten()
var2_c = ran.reshape((dia_c, dia_c), order="C").flatten()
options1_c = np.array([var1_c + i + r_a - r_c for i in range(len(mass)-2*r_a)])
options2_c = np.array([var2_c + i + r_a - r_c for i in range(len(mass)-2*r_a)])

ran = np.tile(np.arange(dia_d), dia_d)
var1_d = ran.reshape((dia_d, dia_d), order="F").flatten()
var2_d = ran.reshape((dia_d, dia_d), order="C").flatten()
options1_d = np.array([var1_d + i + r_a - r_d for i in range(len(mass)-2*r_a)])
options2_d = np.array([var2_d + i + r_a - r_d for i in range(len(mass)-2*r_a)])

# this make_model() function can be used when we do not have a trained model in memory
# or when we want to try out a new architecture. The function returns a model
# model = make_model(r_a, r_b, r_c, r_d, depth)

# the optimizer decides how the weights are udpated
# the parameter given to the optimizer is the learning rate, in this case 3e-4
optimizer = tf.keras.optimizers.Adam(3e-4)

# configuring the model
model.compile(optimizer)
model.summary()

# with neural networks, one epoch means going through the entire dataset once
# since we do not have that amount of time, we define an epoch as being 100.000
# entries from the dataset
epoch_size = 100_000

# the animator shows the progress of the neural network, by graphing the value of the MSE
an = animator.animator()
for _ in range(10000):

    # selects 100.000 random entries from the dataset
    choices = np.random.choice(np.arange(len(indices)), epoch_size, False)
    choices = indices[choices]
    
    # to stabilize training we update the weights by calculating the average desired change over 64 entries
    batch_size = 64
    
    # the amount of batches is determined by how many batches can fit into the size of one epoch. The // operator divides and rounds down.
    batches = epoch_size//batch_size
    
    # an array to keep track of the loss over time
    loss_arr = np.zeros(batches)
    for batch in tqdm.trange(batches):
        # select the dataset entries belonging to this batch
        batch_choices = choices[batch*batch_size:(batch+1)*batch_size, :]

        # these four variables grab the context from the grid with decreasing radii
        net_input_a = mass[options1_a[batch_choices[:, 0] - r_a], options2_a[batch_choices[:, 1]-r_a]].reshape(-1, dia_a, dia_a, depth).astype(np.float32)
        net_input_b = mass[options1_b[batch_choices[:, 0] - r_a], options2_b[batch_choices[:, 1] - r_a]].reshape(-1, dia_b, dia_b, depth).astype(np.float32)
        net_input_c = mass[options1_c[batch_choices[:, 0] - r_a], options2_c[batch_choices[:, 1] - r_a]].reshape(-1, dia_c, dia_c, depth).astype(np.float32)
        net_input_d = mass[options1_d[batch_choices[:, 0] - r_a], options2_d[batch_choices[:, 1] - r_a]].reshape(-1, dia_d, dia_d, depth).astype(np.float32)

        # the desired output, the true convergence, is taken from the convergence grid
        net_output = conv[batch_choices[:, 0], batch_choices[:, 1]].reshape(-1, depth).astype(np.float32)
        with tf.GradientTape() as tape:
            # the model predicts the convergence by processing the input
            output = model([net_input_a, net_input_b, net_input_c, net_input_d])
            
            # the mean square error is computed
            loss = (net_output - output)**2
            
            # because most of the pixels do not contain a value for convergence,
            # as there are 5 times as many pixels as galaxies in the dataset,
            # we have to remove the pixels without convergence from the loss, as to not bias our network
            loss = loss * (net_output != 0)
            loss = tf.reduce_sum(loss, axis=-1)
            
        # lets tensorflow calculate and process the gradients with respect to the loss
        train_vars = model.trainable_variables
        grads = tape.gradient(loss, train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))
        
        # saves the loss, so it can later be displayed to show how good the network is doing
        loss_arr[batch] = np.sum(loss)/np.count_nonzero(loss)
        
    # show the average loss over the entire epoch
    loss = np.mean(loss_arr)
    print("loss:", loss)
    
    # makes a graph of the loss, to show whether the model is still improving
    an.push(loss)
    tf.keras.models.save_model(model, "/home/ignace/tensorTestModels/please_just_work_for_once_2D_v2")

