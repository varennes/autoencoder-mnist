# Autoencoder Module file
# includes class and functions common to AE

import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model, Sequential, model_from_json

class AE_model:
    def __init__(self):
        self.model = Sequential()

    def train(self, x_train, x_validation, n_epoch, n_batch):
        self.history = self.model.fit( x_train, x_train,
                                       epochs=n_epoch, batch_size=n_batch,
                                       shuffle=True,
                                       validation_data=(x_validation, x_validation))

    def load_architecture(self, arch_file):
        with open( arch_file, 'r') as f:
            self.model = model_from_json( f.read())

    def load_weights(self, weights_file):
        self.model.load_weights( weights_file)

    def make_ae_model(self, input_shape):
        self.model = Sequential()
        self.model.add( Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=input_shape))
        self.model.add( MaxPooling2D(pool_size=(2, 2)))
        self.model.add( Conv2D(16, (5, 5), activation='relu', padding='same'))
        self.model.add( MaxPooling2D(pool_size=(2, 2)))
        self.model.add( Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add( MaxPooling2D(pool_size=(2, 2)))
        # decoder
        self.model.add( Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add( UpSampling2D((2,2)))
        self.model.add( Conv2D(16, (5, 5), activation='relu', padding='same'))
        self.model.add( UpSampling2D((2,2)))
        self.model.add( ZeroPadding2D( padding=(1,1)))
        self.model.add( Conv2D(16, (5, 5), activation='relu', padding='same'))
        self.model.add( UpSampling2D((2,2)))
        self.model.add( Conv2D( 1, (3, 3), activation='relu', padding='same'))

    def plot_ae_results(self, x, image_filename, image_shape=(28,28)):
        x_reconstruction = self.model.predict(x)
        n = len(x)
        fig, ax = plt.subplots( 2, n, figsize=(20, 4))
        for i, subplot in enumerate(ax[0]):
            subplot.imshow( x[i].reshape( image_shape[0], image_shape[1]), cmap='gray')
            subplot.axis('off')
        for i, subplot in enumerate(ax[1]):
            subplot.imshow( x_reconstruction[i].reshape( image_shape[0], image_shape[1]), cmap='gray')
            subplot.axis('off')
        fig.suptitle('1st row: Original | 2nd row: Reconstruction', fontsize=16);
        fig.savefig( image_filename)
