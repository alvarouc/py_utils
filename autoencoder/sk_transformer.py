from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .autoencoder import build_autoencoder, build_vae


class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, epochs=1000, batch_size=100, encoding_dim=2,
                 optimizer='adam', verbose=False,
                 *args, **kwargs):
        super(BaseEncoder).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.optimizer = optimizer
        self.verbose = verbose
        self.sc = MinMaxScaler()

    def fit(self, X, y=None):
        self.Xs = self.sc.fit_transform(X)
        return self

    def transform(self, X, y=None):
        Xs = self.sc.transform(X)
        Xe = self.encoder.predict(Xs, verbose=self.verbose,
                                  batch_size=self.batch_size)
        return Xe

    def inverse_transform(self, X):
        Xp = self.decoder.predict(X, verbose=self.verbose,
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

    def fit(self, X, y=None):
        super(AutoEncoder, self).fit(X)
        self.ae, self.encoder, self.decoder =\
            build_autoencoder(self.Xs.shape[1],
                              layers_dim=self.layers_dim,
                              optimizer=self.optimizer)
        self.ae.fit(self.Xs, self.Xs,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    shuffle=True, verbose=self.verbose)
        return self


class VariationalAutoEncoder(BaseEncoder):
    def __init__(self, layer_dim=100,
                 *args, **kwargs):
        super(VariationalAutoEncoder, self).__init__(*args, **kwargs)
        self.layer_dim = layer_dim

    def fit(self, X, **kwargs):
        super(VariationalAutoEncoder, self).fit(X, **kwargs)

        self.vae, self.encoder, self.decoder =\
            build_vae(self.Xs.shape[1], encoding_dim=self.encoding_dim,
                      latent_dim=self.layer_dim, batch_size=self.batch_size,
                      optimizer=self.optimizer)
        self.vae.fit(self.Xs, self.Xs, batch_size=self.batch_size,
                     epochs=self.epochs,
                     shuffle=True, verbose=False)
        return self
