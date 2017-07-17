import pytest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import sys
sys.path.append("../")

from autoencoder import run_vae
print(sys.path)
X, y = make_blobs(n_samples=1000, n_features=30, centers=3,
                  random_state=1988)


@pytest.mark.medium
def test_vae():
    X2, _, _ = run_vae(X, verbose=0, layers_dim=[20, 2],
                       batch_size=100)
    si = silhouette_score(X2, y)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.savefig('test_vae.png')
    assert si > 0.8, 'Score {} < 0.8'.format(si)
    print('Silhouette score: {}'.format(si))


def test_vae_2():
    X2, _, _ = run_vae(X, verbose=0, layers_dim=[20, 20,  2],
                       batch_size=128)
    si = silhouette_score(X2, y[:X2.shape[0]])
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=y[:X2.shape[0]])
    plt.savefig('test_vae2.png')
    assert si > 0.8, 'Score {} < 0.8'.format(si)
    print('Silhouette score: {}'.format(si))
