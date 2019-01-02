# Autoencoder development

import numpy as np
import matplotlib.pyplot as plt

from ae_module import AE_model, add_noise_uni

from keras.datasets import mnist

if __name__ == '__main__':
    # load and prep MNIST data
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test  = x_test.astype( 'float32') / 255.
    x_train = np.reshape( x_train, (len(x_train), 28, 28, 1))
    x_test  = np.reshape( x_test,  (len(x_test), 28, 28, 1))
    input_shape = x_train.shape[1:]

    # create AE model instance, load weights
    ae1 = AE_model()
    ae1.load_architecture( 'ae1_arch.json')
    ae1.load_weights( 'ae1_weights.h5')

    # add noise to data
    noise_factor = 0.5
    x_train_noise = add_noise_uni( x_train, noise_factor)
    x_test_noise  = add_noise_uni( x_test,  noise_factor)

    print ae1.model.summary()
    ae1.model.compile( optimizer='adadelta', loss='binary_crossentropy')

    history_ae = ae1.model.fit( x_train_noise, x_train,
                                epochs=200,
                                batch_size=256,
                                shuffle=True,
                                validation_data=(x_test_noise, x_test))

    ae1.plot_ae_results( x_test_noise[:10], 'aeNoise1_test1.png')

    # save model to file
    ae1.model.save_weights( 'aeNoise1_weights.h5')
    with open( 'aeNoise1_arch.json', 'w') as f:
        f.write( ae1.model.to_json())
