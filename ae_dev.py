# Autoencoder development
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model, Sequential, model_from_json

class AE_model:
    def __init__(self):
        self.model = Sequential()

    def load_architecture(self, arch_file):
        with open( arch_file, 'r') as f:
            self.model = model_from_json( f.read())

    def load_weights(self, weights_file):
        self.model.load_weights( weights_file)
