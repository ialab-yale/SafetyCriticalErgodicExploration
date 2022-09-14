import jax.numpy as np

def dist_func(x):
    d1 = 0.1-np.linalg.norm(x[0,:] - x[1,:])
    d2 = 0.1-np.linalg.norm(x[1,:] - x[2,:])
    d3 = 0.1-np.linalg.norm(x[0,:] - x[2,:])
    dist = [d1.flatten(), d2.flatten(), d3.flatten()]
    return np.array(dist)

def sdf2cbf(f, constr):
    return lambda x, u, alpha: constr(f(x,u)) - (1.-alpha) * constr(x)