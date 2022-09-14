import jax.numpy as np

class ErgodicMetric(object):
    '''Class for constructing the ergodic metric'''
    def __init__(self, basis) -> None:
        self.basis = basis
        self.lamk = (1.+np.linalg.norm(basis.k_list/np.pi,axis=1)**2)**(-(basis.n+1)/2.)
        # lamk = np.exp(-0.8 * np.linalg.norm(k, axis=1))
        # lamk = np.ones((len(k), 1))
   
    def __call__(self, ck, phik):
        return np.sum(self.lamk * (ck - phik)**2)
    
    def eval_delta_cost(self, dc):
        return np.mean(np.square(dc) @ self.lamk)
    
    def construct_erg_dyn(self, robot_model):
        def f_erg(x, c, phik):
            dcdt = self.basis.fk_vmap(x[:2])/self.basis.hk_list - phik
            return c + robot_model.dt *dcdt
        return f_erg