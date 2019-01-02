# Autoencoder development

import numpy as np
import matplotlib.pyplot as plt

from ae_module import AE_model

from keras.datasets import mnist

if __name__ == '__main__':
    # load and prep MNIST data
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test  = x_test.astype( 'float32') / 255.
    x_train = np.reshape( x_train, (len(x_train), 28, 28, 1))
    x_test  = np.reshape( x_test,  (len(x_test), 28, 28, 1))
    input_shape = x_train.shape[1:]

    # create AE model instance
    ae1 = AE_model()
    ae1.make_ae_model( input_shape)
    print ae1.model.summary()
    ae1.model.compile( optimizer='adadelta', loss='binary_crossentropy')
