from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from logger import make_logger
from multigpu import make_parallel
# from keras.callbacks import TensorBoard

log = make_logger('autoencoder')


def build_autoencoder(input_dim, ngpu=1, layers_dim=[100, 10, 10],
                      activations=['tanh', 'tanh'],
                      inits=['glorot_uniform', 'glorot_normal'],
                      optimizer='adam',
                      drop=0.1, l2=1e-5,
                      loss='mse'):

    input_row = Input(shape=(input_dim,))

    for n, layer_dim in enumerate(layers_dim):
        if n == 0:
            encoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(input_row)
            if drop > 0:
                encoded = Dropout(drop)(encoded)
        elif n < (len(layers_dim) - 1):
            encoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(encoded)
            if drop > 0:
                encoded = Dropout(drop)(encoded)
        else:
            encoded = Dense(layer_dim, activation=activations[0],
                            activity_regularizer=regularizers.l2(l2),
                            kernel_initializer=inits[0])(encoded)

    encoder = Model(input_row, encoded)
    # if ngpu > 1:
    #  encoder = make_parallel(encoder, ngpu)
    for n, layer_dim in enumerate(reversed(layers_dim[:-1])):
        if n == 0:
            decoded = Dense(layer_dim, activation=activations[0],
                            kernel_initializer=inits[0])(encoded)
            if drop > 0:
                decoded = Dropout(drop)(decoded)
        else:
            decoded = Dense(layer_dim, activation=activations[0],
                            kernel_regularizer=regularizers.l2(l2),
                            kernel_initializer=inits[0])(decoded)
            if drop > 0:
                decoded = Dropout(drop)(decoded)

    decoded = Dense(input_dim, activation=activations[1],
                    kernel_regularizer=regularizers.l2(l2),
                    kernel_initializer=inits[1])(decoded)

    autoencoder = Model(input_row, decoded)
    log.info(autoencoder.summary())

    if ngpu > 1:
        autoencoder = make_parallel(autoencoder, ngpu)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder, encoder


def run_ae(X, epochs=100, batch_size=128, verbose=0,  **kwargs):
    log.info('Standarize')
    if X.dtype == 'bool':
        log.info('Boolean data detected')
        loss = 'binary_crossentropy'
        Xs = X
    else:
        # Standarize
        ptp = X.ptp(axis=0)
        ptp[ptp == 0] = 1
        Xs = (X - X.min(axis=0)) / ptp
        loss = 'mse'
        log.info('Done. AE Loss: {}'.format(loss))

    # AE
    ae_args = {'layers_dim': [100, 50,
                              min(max(X.shape[1] // 10, 5), 25)],
               'inits': ['glorot_normal', 'glorot_uniform'],
               'activations': ['tanh', 'sigmoid'], 'l2': 0,
               'optimizer': 'adagrad'}
    ae_args.update(kwargs)
    log.info('Training Autoencoder')
    ae, encoder = build_autoencoder(Xs.shape[1], **ae_args)
    ae.fit(Xs, Xs, batch_size=batch_size, epochs=epochs,
           shuffle=True, verbose=verbose)
    log.info('Encoding')
    Xp = encoder.predict(Xs, verbose=False)
    log.info('Computing reconstruction loss')
    X2 = ae.predict(Xs)
    error = ((X2 - Xs)**2).mean(axis=0)
    log.info('Done. Loss %s', ae.evaluate(Xs, Xs))
    return Xp, error, encoder
