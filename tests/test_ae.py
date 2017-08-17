import pytest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error as mse
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from autoencoder import run_ae, AutoEncoder

X, y = make_blobs(n_samples=1000, n_features=30, centers=3,
                  random_state=1988)


def test_ae():
    X2 = run_ae(X, verbose=1, layers_dim=[20, 2],
                compute_error=False)
    si = silhouette_score(X2, y)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.savefig('ae.png')
    assert si > 0.8, 'Score {} < 0.8'.format(si)
    print('Silhouette score: {}'.format(si))


def test_ae_2():
    X2 = run_ae(X, verbose=1, layers_dim=[20, 20,  2])
    si = silhouette_score(X2, y)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.savefig('ae2.png')
    assert si > 0.8, 'Score {} < 0.8'.format(si)
    print('Silhouette score: {}'.format(si))


def test_sk_ae():
    sk = AutoEncoder(layers_dim=[20, ], encoding_dim=2)
    sk3 = AutoEncoder(layers_dim=[20, ], encoding_dim=10)

    X2 = sk.fit_transform(X)
    X3 = sk3.fit_transform(X)

    si = silhouette_score(X2, y)
    assert si > 0.8, 'Score {} < 0.8'.format(si)
    print('X2 Silhouette score: {}'.format(si))

    Xr = sk.inverse_transform(X2)
    error = mse(X, Xr)
    print('Error : {}'.format(error))
    Xr3 = sk3.inverse_transform(X3)
    error3 = mse(X, Xr3)
    print('Error3 : {}'.format(error3))
    assert error3 < error, 'Bigger network has more error'
