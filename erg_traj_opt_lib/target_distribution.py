import jax.numpy as np
from jax import vmap

class TargetDistribution(object):
    def __init__(self) -> None:
        self.n      = 2
        self.domain = np.meshgrid(                                      # the domain is from 0,1
            *[np.linspace(0,1,100)]*self.n
        )
        self._s     = np.stack([X.ravel() for X in self.domain]).T      # _s sets up the full spatial domain
        self.evals  = (
            vmap(self.p)(self._s) , self._s                             # evals is the distribution mapped to the spatial domain?
        )

    # # uniform distribution
    # def p(self, x):
    #     return 1
    
    # bimodal distribution
    def p(self, x):
        return np.exp(-60.5 * np.sum((x[:2] - 0.2)**2)) \
                    + np.exp(-60.5 * np.sum((x[:2] - 0.75)**2)) \

    def update(self):
        pass

if __name__=='__main__':
    import matplotlib.pyplot as plt
    
    target_distr = TargetDistribution()
    X, Y         = target_distr.domain
    _s           = target_distr._s
    vals         = target_distr.evals[0]

    plt.figure()
    plt.contourf(X, Y, vals.reshape(X.shape), cmap='plasma')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.savefig('test_fig.png')

