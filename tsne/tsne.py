from logger import make_logger
from sklearn.manifold import TSNE

log = make_logger('TSNE')


def run_tsne(Xp, **kwargs):
    # TSNE
    log.info('Running TSNE on {} samples with params: {}'.format(
        Xp.shape[0], kwargs))
    Xt = TSNE(**kwargs).fit_transform(Xp)
    log.info('Done')
    return Xt
