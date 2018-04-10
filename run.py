import numpy as np
from sklearn.model_selection import train_test_split

data = np.load('data_small_0eta1.npy')
X = data[:,(0,7)]
y = data[:,(1,4)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from keras.layers import Input, Dense
from keras.models import Model

input_dim = X_train.shape[1]

input_layer = Input(shape = (input_dim, ))
encoder = Dense(32, activation='relu')(input_layer)
encoder = Dense(16, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs = input_layer, outputs = decoder)

nb_epoch = 20
batch_size = 2048

autoencoder.compile(optimizer = 'adam',
                    loss = 'kullback_leibler_divergence')

from keras.callbacks import ModelCheckpoint, TensorBoard

checkpointer = ModelCheckpoint(filepath = 'model.h5',
                               save_best_only = True)
tensorboard = TensorBoard(log_dir = './logs',
                          histogram_freq = 0,
                          write_graph = True,
                          write_images = True)

history = autoencoder.fit(X_train, X_train,
                          epochs = nb_epoch,
                          batch_size = batch_size,
                          shuffle = True,
                          validation_data = (X_test, X_test),
                          callbacks = [checkpointer, tensorboard]).history
