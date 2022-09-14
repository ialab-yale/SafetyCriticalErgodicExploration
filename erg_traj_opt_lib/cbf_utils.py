import jax.numpy as np

def dist(x1, x2, r):
    return r - np.linalg.norm(x1-x2)

def sdf2cbf(f, constr):
    return lambda x, u, alpha: constr(f(x,u)) - (1.-alpha) * constr(x)