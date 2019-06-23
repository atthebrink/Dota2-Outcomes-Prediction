from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model

import keras
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split

import pandas as pd

initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)


class AAN:
    def __init__(self, input_dim, latent_dim=2, lr=0.001, lamda=0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.lamda = lamda
        self.encoder = self._genEncoderModel(input_dim, latent_dim)
        self.decoder = self._getDecoderModel(input_dim, latent_dim)
        self.discriminator = self._getDescriminator(latent_dim)

        autoencoder_input = keras.Input(shape=(input_dim,))
        generator_input = keras.Input(shape=(input_dim,))

        self.autoencoder = keras.Model(autoencoder_input, self.decoder(self.encoder(autoencoder_input)))

        self.discriminator.compile(optimizer=keras.optimizers.Adam(lr),
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

        self.autoencoder.compile(optimizer=keras.optimizers.Adam(lamda * lr),
                                 loss='mse')

        self.discriminator.trainable = False
        self.encoder_discriminator = keras.Model(generator_input, self.discriminator(self.encoder(generator_input)))
        self.encoder_discriminator.compile(optimizer=keras.optimizers.Adam(lr),
                                           loss='binary_crossentropy',
                                           metrics=['accuracy'])
        # print("Encoder Architecture")
        # print(self.encoder.summary())
        # print("Decoder Architecture")
        # print(self.decoder.summary())
        # print("Discriminator Architecture")1
        # print(self.discriminator.summary())
        # print("Autoencoder Architecture")
        # print(self.autoencoder.summary())
        # print("Generator  Architecture")
        # print(self.encoder_discriminator.summary())

    def _genEncoderModel(self, input_dim, latent_dim):
        """ Build Encoder Model Based on Paper Configuration
        Args:
            img_shape (tuple) : shape of input image
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        encoder = keras.Sequential([
            keras.layers.Dense(1000, activation='relu', input_dim=input_dim),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(latent_dim, activation=None)
        ])
        return encoder

    def _getDecoderModel(self, output_dim, latent_dim):
        """ Build Decoder Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
            img_shape (tuple) : shape of target images
        Return:
            A sequential keras model
        """
        decoder = keras.Sequential([
            keras.layers.Dense(1000, activation='relu', input_dim=latent_dim),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(output_dim, activation='sigmoid')
        ])
        return decoder

    def _getDescriminator(self, latent_dim):
        """ Build Descriminator Model Based on Paper Configuration
        Args:
            encoded_dim (int) : number of latent variables
        Return:
            A sequential keras model
        """
        discriminator = keras.Sequential([
            keras.layers.Dense(1000, activation='relu', input_dim=latent_dim),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        return discriminator

    def save(self, epochs=0):
        self.encoder.save("model/encoder_" + str(epochs) + ".h5")
        self.decoder.save("model/decoder_" + str(epochs) + ".h5")
        self.discriminator.save("model/discriminator_" + str(epochs) + ".h5")
        self.autoencoder.save("model/autoencoder_" + str(epochs) + ".h5")
        self.encoder_discriminator.save("model/encoder_discriminator_" + str(epochs) + ".h5")

    def load(self, epochs=0):
        self.encoder = keras.models.load_model("model/encoder_" + str(epochs) + ".h5")
        self.decoder = keras.models.load_model("model/decoder_" + str(epochs) + ".h5")
        self.discriminator = keras.models.load_model("model/discriminator_" + str(epochs) + ".h5")
        self.autoencoder = keras.models.load_model("model/autoencoder_" + str(epochs) + ".h5")
        self.encoder_discriminator = keras.models.load_model("model/encoder_discriminator_" + str(epochs) + ".h5")

    def train(self, X, weight, batch_size=64, epochs=50, save_interval=10):
        half_batch = int(batch_size / 2)
        for epoch in range(epochs):
            # ---------------Train Discriminator -------------
            # Select a random half batch of images
            idx = np.random.randint(0, X.shape[0], half_batch)
            samples = X[idx]
            # Generate a half batch of latent
            latent_fake = self.encoder.predict(samples)
            latent_real = 5 * np.random.normal(size=(half_batch, latent_fake.shape[1]))
            dis_X = np.concatenate([latent_fake, latent_real], axis=0)
            dis_Y = np.concatenate([np.zeros((half_batch, 1)), np.ones((half_batch, 1))], axis=0)
            # Train the discriminator
            discriminator_loss = self.discriminator.train_on_batch(dis_X, dis_Y)

            idx = np.random.randint(0, X.shape[0], batch_size)
            samples = X[idx]
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.ones((batch_size, 1))

            # Train the autoencoder reconstruction
            autoencoder_loss = self.autoencoder.train_on_batch(samples, samples, sample_weight=weight[idx])

            # Train generator
            generator_loss = self.encoder_discriminator.train_on_batch(samples, valid_y, sample_weight=weight[idx])
            # generator_loss = [0.0,0.0]
            # Plot the progress
            print("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch, discriminator_loss[0], 100 * discriminator_loss[1],
                                                                         generator_loss[1], autoencoder_loss))
            if epoch % save_interval == 0:
                self.save(epoch)

        self.save(epochs)

    def retrain(self, X, batch_size=64, epochs=50):
        for epoch in range(epochs):
            idx = np.random.randint(0, X.shape[0], batch_size)
            samples = X[idx]
            # Generator wants the discriminator to label the generated representations as valid
            valid_y = np.zeros((batch_size, 1))
            weights = np.ones(samples.shape[0]) - 2
            # Train the autoencoder reconstruction
            autoencoder_loss = self.autoencoder.train_on_batch(samples, samples, sample_weight=weights)

            # Train generator
            generator_loss = self.encoder_discriminator.train_on_batch(samples, valid_y, sample_weight=weights)
            # generator_loss = [0.0,0.0]
            # Plot the progress
            print("%d [G acc: %f, mse: %f]" % (epoch, generator_loss[1], autoencoder_loss))


def load_data(file_name, min_mmr=0, gamemode=22):
    raw = np.array(pd.read_csv(file_name))
    raw = raw[raw[:, 4] > min_mmr]
    raw = raw[raw[:, 6] == gamemode]

    data = np.zeros((raw.shape[0], 114 * 2 + 1), dtype=int)
    label = np.array(raw[:, 1], dtype=int)

    for i, rec in enumerate(raw):
        # radiant win
        data[i, 0] = int(rec[1])
        # radiant team picks
        for t in map(int, rec[2].split(',')):
            data[i, t] = 1
        # dire team picks
        for t in map(int, rec[3].split(',')):
            data[i, t + 114] = 1

    return data


def pretrain(X, Y, epochs=5):
    train_data, test_data, train_label, test_label = train_test_split(X, Y, train_size=0.1, random_state=42)
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_label, epochs=epochs, batch_size=64)
    loss, acc = model.evaluate(test_data, test_label)
    print("pre-training: loss:%f, acc:%f" % (loss, acc))
    label = model.predict(X)
    weight = label[np.arange(label.shape[0]), Y]

    return weight


if __name__ == '__main__':
    data = load_data("706e_train_dataset.csv", min_mmr=0)
    print(data.shape)
    Y = data[:, 0]
    X = data[:, 1:]
    print(X.shape)
    weight = pretrain(X, Y, epochs=5)

    aan = AAN(input_dim=data.shape[1], latent_dim=10)
    aan.train(data, epochs=3000, save_interval=100, weight=weight)
    aan.save(epochs=3000)

    for iter in range(5):
        print("iteration: " + str(iter))
        print(data.shape)
        print("generate latent code")
        reconstructed = aan.autoencoder.predict(data)
        # calculate mse for every row
        i = 0
        err = np.array([])
        while i < data.shape[0]:
            err_i = (np.array(data[i:i + 64] - reconstructed[i:i + 64]) ** 2).mean(axis=1)
            err = np.concatenate([err, err_i], axis=0)
            i += 64
        likelihood = aan.encoder_discriminator.predict(data)
        print("identify anomalies candidate ")
        err = np.squeeze(err)
        likelihood = np.squeeze(likelihood)

        idx1 = np.argsort(err)
        idx2 = np.argsort(likelihood)
        i = int(err.shape[0] * 0.4)
        j = int(likelihood.shape[0] * 0.4)
        idx1 = idx1[-i:]
        idx2 = idx2[:j]
        mask = np.ones(data.shape[0])
        mask = np.array(mask, dtype=bool)
        idx1 = set(idx1)

        for i in idx2:
            if i in idx1:
                mask[i] = False

        anomalies = data[np.invert(mask)]
        data = data[mask]

        print(data.shape)
        np.savetxt("data_filtered_" + str(iter + 1) + ".csv", data, delimiter=',', fmt="%d")
        aan.retrain(anomalies)
