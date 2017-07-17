import pytest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import sys
sys.path.append("../")

from autoencoder import run_ae

X, y = make_blobs(n_samples=1000, n_features=30, centers=3,
                  random_state=1988)


@pytest.mark.medium
def test_ae():
    X2, _, _ = run_ae(X, verbose=1, layers_dim=[20, 2])
    si = silhouette_score(X2, y)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.savefig('test_ae.png')
    assert si > 0.8, 'Score {} < 0.8'.format(si)
    print('Silhouette score: {}'.format(si))


def test_ae_2():
    X2, _, _ = run_ae(X, verbose=1, layers_dim=[20, 20,  2])
    si = silhouette_score(X2, y)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.savefig('test_ae2.png')
    assert si > 0.8, 'Score {} < 0.8'.format(si)
    print('Silhouette score: {}'.format(si))
