import jax.numpy as np

class SingleIntegrator(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 2
        self.m = 2
        def f(x, u):
            B = np.array([
                [1.,0.],
                [0.,1.]
            ])
            # B = np.array([
            #     [np.cos(x[2]), 0.,],
            #     [np.sin(x[2]), 0.],
            #     [0., 1.]
            # ])
            return x + self.dt*B@u
        self.f = f

class KinematicUnicycle(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 3
        self.m = 2
        def f(x, u):
            B = np.array([
                [np.cos(x[2]), 0.,],
                [np.sin(x[2]), 0.],
                [0., 1.]
            ])
            return x + self.dt*B@u
        self.f = f

class Single3DoFIntegrator(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 3
        self.m = 3
        def f(x, u):
            return x + self.dt*u
        self.f = f