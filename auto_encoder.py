import os
import numpy as np

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential

class AutoEncoder:
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        self.model = self._auto_encoder()

    def _encoder(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
            input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        return model

    def _decoder(self):
        model = Sequential()
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same',
            input_shape=(4, 4, 8)))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
        return model

    def _auto_encoder(self):
        model = Sequential()
        model.add(self.encoder)
        model.add(self.decoder)
        model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, data, batch_size=32, epochs=300, validation_data=None):
        self.model.fit(data, data, epochs=epochs, batch_size=batch_size,
                validation_data=validation_data,
                shuffle=True)

    def save(self):
        if not os.path.exists('./saved_models'):
            os.mkdir('./saved_models')
        else:
            self.encoder.save('./saved_models/encoder_weights.h5')
            self.decoder.save('./saved_models/decoder_weights.h5')
            self.model.save('./saved_models/ae_weights.h5')

if __name__ == '__main__':
    from keras.datasets import mnist
    import matplotlib.pyplot as plt

    (x_train, _), (x_test, _) = mnist.load_data()

    # noramlise pixel values
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # reshape input tensors
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],
        x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],
        x_test.shape[2], 1))

    # train auto encoder
    auto_encoder = AutoEncoder(input_shape=x_train[0].shape)
    auto_encoder.fit(x_train, epochs=50, batch_size=128, validation_data=(x_test, x_test))
    auto_encoder.save()

    # run encoded/decoded predictions
    encoded_imgs = auto_encoder.encoder.predict(x_test)
    decoded_imgs = auto_encoder.decoder.predict(encoded_imgs)

    # show original vs decoded images
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
