from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras import metrics
from keras.engine.topology import Layer
from keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import warnings
import pdb


def build_vae(input_dim,
              latent_dim=100,
              encoding_dim=10,
              activations=['relu', 'sigmoid'],
              inits=['glorot_uniform', 'glorot_normal'],
              optimizer='adam', batch_size=512,
              epsilon_std=0.1):

    x = Input(batch_shape=(batch_size, input_dim))
    h = Dense(latent_dim, activation=activations[0],
              kernel_initializer=inits[0])(x)

    z_mean = Dense(encoding_dim)(h)
    z_log_var = Dense(encoding_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, encoding_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    decoder_h = Dense(latent_dim, activation=activations[0],
                      kernel_initializer=inits[0])
    decoder_mean = Dense(input_dim, activation=activations[1],
                         kernel_initializer=inits[1])
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = input_dim * \
                metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var -
                                    K.square(z_mean) - K.exp(z_log_var),
                                    axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            return x

    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)
    vae.compile(optimizer=optimizer, loss=None)
    encoder = Model(x, z_mean)

    # build a generator that can sample from the learned distribution
    decoder_input = Input(shape=(encoding_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    decoder = Model(decoder_input, _x_decoded_mean)

    return vae, encoder, decoder


def build_autoencoder(input_dim, layers_dim=[100, 10, 10],
                      activations=['tanh', 'sigmoid'],
                      inits=['glorot_normal', 'glorot_normal'],
                      optimizer='adam',
                      l2=0,
                      loss='mse'):

    input_row = Input(shape=(input_dim,))

    for n, layer_dim in enumerate(layers_dim):
        if n == 0:
            encoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(input_row)
        elif n < (len(layers_dim) - 1):
            encoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(encoded)
        else:
            encoded = Dense(layer_dim, activation=activations[0],
                            activity_regularizer=regularizers.l2(l2),
                            kernel_initializer=inits[0])(encoded)

    encoder = Model(input_row, encoded)
    # create a placeholder for an encoded (n-dimensional) input
    encoded_input = Input(shape=(layers_dim[-1],))

    for n, layer_dim in enumerate(reversed(layers_dim[:-1])):
        if n == 0:
            decoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(encoded)
        else:
            decoded = Dense(layer_dim, activation=activations[0],
                            kernel_regularizer=regularizers.l2(l2),
                            kernel_initializer=inits[0])(decoded)

    decoded = Dense(input_dim, activation=activations[1],
                    kernel_regularizer=regularizers.l2(l2),
                    kernel_initializer=inits[1])(decoded)

    autoencoder = Model(input_row, decoded)

    encoded_input = Input(shape=(layers_dim[-1],))
    for n, layer_dim in enumerate(reversed(layers_dim)):
        if n == 0:
            deco = autoencoder.layers[-len(layers_dim)](encoded_input)
        else:
            deco = autoencoder.layers[-(len(layers_dim) - n)](deco)
    # create the decoder model
    decoder = Model(encoded_input, deco)

    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder, encoder, decoder


def standard(X):
    if X.dtype == 'bool':
        loss = 'binary_crossentropy'
        Xs = X
    else:
        # Standarize
        ptp = X.ptp(axis=0)
        ptp[ptp == 0] = 1
        Xs = (X - X.min(axis=0)) / ptp
        loss = 'mse'
    return Xs, loss


def run_ae(X, epochs=100, batch_size=128, verbose=False,
           compute_error=False, **kwargs):

    Xs, loss = standard(X)
    # AE
    if verbose:
        print('Training Autoencoder')
    ae, encoder, decoder = build_autoencoder(Xs.shape[1], **kwargs)
    ae.fit(Xs, Xs, batch_size=batch_size, epochs=epochs, shuffle=True,
           verbose=verbose, callbacks=[
               EarlyStopping(monitor='loss',
                             min_delta=0, patience=50, verbose=1,
                             mode='auto'),
               TensorBoard(log_dir='/tmp/autoencoder')])
    Xp = encoder.predict(Xs, verbose=False)

    if compute_error:
        X2 = ae.predict(Xs, verbose=False)
        error = ((X2 - Xs)**2).mean(axis=0)
        if verbose:
            print('Loss {:.2e}'.format(
                ae.evaluate(Xs, Xs, verbose=verbose)))
        return Xp, error, encoder
    else:
        return Xp


def run_vae(X, epochs=100, batch_size=128, verbose=False,
            compute_error=False, **kwargs):

    remove = X.shape[0] % batch_size
    if remove != 0:
        warnings.warn(
            'Batch size ({}) is not multiple of the number of samples ({}),\
Ignoring last {} samples'.format(batch_size, X.shape[0], remove))
        X = X[:-remove, :]

    Xs, _ = standard(X)
    kwargs['batch_size'] = batch_size
    # VAE
    vae, encoder, decoder = build_vae(Xs.shape[1], **kwargs)
    if verbose:
        print(vae.summary())
    vae.fit(Xs, Xs, batch_size=batch_size, epochs=epochs,
            shuffle=True, verbose=verbose,
            callbacks=[
                EarlyStopping(monitor='loss', min_delta=0,
                              patience=50, verbose=0, mode='auto'),
                TensorBoard(log_dir='/tmp/autoencoder')])
    Xp = encoder.predict(Xs, verbose=False, batch_size=batch_size)

    if compute_error:
        X_latent = encoder.predict(Xs, verbose=False, batch_size=batch_size)
        X2 = decoder.predict(X_latent, verbose=False, batch_size=batch_size)
        error = ((X2 - Xs)**2).mean(axis=0)
        if verbose:
            print('MSE {:.2e}'.format(np.mean(error)))
        return Xp, error, encoder, decoder
    else:
        return Xp
