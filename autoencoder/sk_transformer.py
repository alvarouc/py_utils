import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from .autoencoder import build_autoencoder, build_vae
import pdb


class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, epochs=1000, batch_size=100, encoding_dim=2,
                 *args, **kwargs):
        super(BaseEncoder).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.sc = MinMaxScaler()

    def fit(self, X, epochs=100):
        self.Xs = self.sc.fit_transform(X)
        return self

    def transform(self, X, y=None, verbose=False):
        Xs = self.sc.transform(X)
        Xe = self.encoder.predict(Xs, verbose=verbose,
                                  batch_size=self.batch_size)
        return Xe

    def inverse_transform(self, X):
        Xp = self.decoder.predict(X, verbose=False,
                                  batch_size=self.batch_size)
        Xr = self.sc.inverse_transform(Xp)
        return Xr

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)


class AutoEncoder(BaseEncoder):
    def __init__(self, layers_dim=[100, 10],
                 *args, **kwargs):
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.layers_dim = layers_dim
        self.layers_dim.append(self.encoding_dim)

    def fit(self, X, verbose=True, **kwargs):
        super(AutoEncoder, self).fit(X, **kwargs)
        self.ae, self.encoder, self.decoder =\
            build_autoencoder(self.Xs.shape[1],
                              layers_dim=self.layers_dim)
        self.ae.fit(self.Xs, self.Xs,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=True, verbose=verbose)
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
