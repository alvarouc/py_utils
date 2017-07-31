import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from .autoencoder import build_autoencoder, build_vae


class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=100, *args, **kwargs):
        super(BaseEncoder).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.sc = MinMaxScaler()

    def fit(self, X, epochs=100):
        self.Xs = self.sc.fit_transform(X)
        self.epochs = epochs
        return self

    def transform(self, X, y=None):
        Xe = self.encoder.predict(X, verbose=False,
                                  batch_size=self.batch_size)
        return Xe

    def inverse_transform(self, X):
        Xp = self.decoder.predict(X, verbose=False,
                                  batch_size=self.batch_size)
        return self.sc.inverse_transform(Xp)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)


class AutoEncoder(BaseEncoder):
    def __init__(self, encoding_dim=2, layers_dim=[100, 10],
                 *args, **kwargs):
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.encoding_dim = encoding_dim
        self.layers_dim = layers_dim
        self.layers_dim.append(self.encoding_dim)

    def fit(self, X, **kwargs):
        super(AutoEncoder, self).fit(X, **kwargs)
        remove = X.shape[0] % self.batch_size
        if remove != 0:
            warnings.warn(
                'Batch size ({}) is not multiple of the number of samples ({}),\
                Ignoring last {} samples'.format(self.batch_size,
                                                 self.Xs.shape[0],
                                                 remove))
            self.Xs = self.Xs[:-remove, :]

        self.ae, self.encoder, self.decoder =\
            build_autoencoder(self.Xs.shape[1],
                              layers_dim=self.layers_dim)
        self.ae.fit(self.Xs, self.Xs,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=True, verbose=False)
        return self


class VariationalAutoEncoder(BaseEncoder):
    def __init__(self, encoding_dim=2, layer_dim=100,
                 *args, **kwargs):
        super(VariationalAutoEncoder, self).__init__(*args, **kwargs)
        self.encoding_dim = encoding_dim
        self.layer_dim = layer_dim

    def fit(self, X, **kwargs):
        super(VariationalAutoEncoder, self).fit(X, **kwargs)

        self.vae, self.encoder, self.decoder =\
            build_vae(self.Xs.shape[1], encoding_dim=self.encoding_dim,
                      latent_dim=self.layer_dim, batch_size=self.batch_size)
        self.vae.fit(self.Xs, self.Xs, batch_size=self.batch_size,
                     epochs=self.epochs,
                     shuffle=True, verbose=False)
        return self
