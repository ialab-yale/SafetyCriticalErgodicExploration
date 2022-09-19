import jax.numpy as np

class SingleIntegrator(object):
    def __init__(self) -> None:
        self.dt = 0.2
        self.n = 2
        self.m = 2
        B = np.array([
                [1.,0.],
                [0.,1.]
            ])
        def f(x, u):
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

class MultiRobotSingleIntegrator(object):
    def __init__(self, N=4) -> None:
        self.dt = 0.1
        self.n = 2
        self.m = 2
        self.N = N
        B = np.array([
                [1.,0.],
                [0.,1.]
            ])
        def _f(x, u):
            return x + self.dt*B@u
        def f(x1, u1):
            # assumes x1 a Nxn, u1 a Nxm dim
            x2 = []
            for i in range(self.N):
                x2.append(_f(x1[i,:], u1[i,:]))
            return np.stack(x2)
        self.f = f

if __name__=='__main__':
    sys = MultiRobotSingleIntegrator()
    x = np.ones((10,3,2))
    u = np.ones((10,3,2))
    from jax import vmap
    print(vmap(sys.f)(x, u).shape)