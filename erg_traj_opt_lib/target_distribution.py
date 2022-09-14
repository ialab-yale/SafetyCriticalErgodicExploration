import jax.numpy as np
from jax import vmap

class TargetDistribution(object):
    def __init__(self) -> None:
        self.n = 2
        self.domain = np.meshgrid(
            *[np.linspace(0,1)]*self.n
        )
        self._s = np.stack([X.ravel() for X in self.domain]).T
        self.evals = (
            vmap(self.p)(self._s) , self._s
        )

    def p(self, x):
        return 1
        # return np.exp(-60.5 * np.sum((x[:2] - 0.2)**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - 0.75)**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.2, 0.75]))**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.75, 0.2]))**2))

    def update(self):
        pass