### Jax/numpoy imports 
from time import time
import jax
import jax.numpy as np

from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian
from jax.lax import scan
import jax.random as jnp_random
from jax.flatten_util import ravel_pytree
import numpy as onp

### TODO
import pickle as pkl ## <--- this will probably get pushed up to user side

### Local imports
from .solver import AugmentedLagrangian
from .ergodic_metric import ErgodicMetric
from .cbf_utils import sdf2cbf
from .fourier_utils import BasisFunc, get_phik, get_ck
from .target_distribution import TargetDistribution


class MAErgodicTrajectoryOpt(object):
    def __init__(self, robot_model, obstacles,
                        basis=None, time_horizon=500, args=None) -> None:
        self.time_horizon    = time_horizon
        self.robot_model     = robot_model
        self.basis = basis
        self.erg_metric      = ErgodicMetric(self.basis)
        _n, _m, _N = self.robot_model.n, self.robot_model.m, self.robot_model.N
        
        x = np.linspace(args['x0'], args['xf'], time_horizon, endpoint=True)

        u = np.zeros((time_horizon, _N, _m))
        self.init_sol = np.concatenate([x, u], axis=2)
        
        self.def_args = args
        self.obs = obstacles

        self.cbf_consts = []
        for obs in self.obs: 
            self.cbf_consts.append(sdf2cbf(self.robot_model.f, vmap(obs.distance)))
        def _emap(x, args):
            """ Function that maps states to workspace """
            wrksp_bnds = args['wrksp_bnds']
            return np.array([
                (x[0]-wrksp_bnds[0,0])/(wrksp_bnds[0,1]-wrksp_bnds[0,0]), 
                (x[1]-wrksp_bnds[1,0])/(wrksp_bnds[1,1]-wrksp_bnds[1,0])])
        emap = vmap(vmap(_emap, in_axes=(0,None)), in_axes=(0,None))

        def barrier_cost(e):
            """ Barrier function to avoid robot going out of workspace """
            return (np.maximum(0, e-1) + np.maximum(0, -e))**2
        @jit
        def loss(z, args):
            """ Traj opt loss function, not the same as erg metric """
            x, u = z[:, :, :_n], z[:, :, _n:]
            phik = args['phik']
            e = emap(x, args)
            ck = np.mean(vmap(get_ck, in_axes=(1, None))(e, self.basis), axis=0)
            return self.erg_metric(ck, phik) \
                    + 0.1 * np.mean(u**2) \
                    + np.sum(barrier_cost(e))

        def eq_constr(z, args):
            """ dynamic equality constriants """
            x, u = z[:, :, :_n], z[:, :, _n:]
            x0 = args['x0']
            xf = args['xf']
            re = x[1:,:]-vmap(self.robot_model.f)(x[:-1,:], u[:-1,:])
            val= np.concatenate([
                (x[0] - x0).flatten(),
                re.flatten(),
                (x[-1] - xf).flatten()
            ])
            return val

        def ineq_constr(z, args):
            """ control inequality constraints"""
            x, u = z[:, :, :_n], z[:, :, _n:]

            d1 = 0.1-np.linalg.norm(x[:,0,:] - x[:,1,:], axis=1)
            d2 = 0.1-np.linalg.norm(x[:,1,:] - x[:,2,:], axis=1)
            d3 = 0.1-np.linalg.norm(x[:,0,:] - x[:,2,:], axis=1)
            dist = [d1.flatten(), d2.flatten(), d3.flatten()]
            # p = x[:,:2] # extract just the position component of the trajectory
            # obs_val = [vmap(_ob.distance)(p).flatten() for _ob in self.obs]
            obs_val = [vmap(_cbf_ineq, in_axes=(0,0,None))(x, u, args['alpha']).flatten() for _cbf_ineq in self.cbf_consts]

            ctrl_box = [(np.abs(u) - 6.).flatten()]
            _ineq_list = ctrl_box + obs_val + dist
            return np.concatenate(_ineq_list)

        self.eq_constr = eq_constr
        self.ineq_constr = ineq_constr

        self.solver = AugmentedLagrangian(
                                            self.init_sol,
                                            loss, 
                                            eq_constr, 
                                            ineq_constr, 
                                            args, 
                                            step_size=0.01,
                                            c=0.1
                    )
        @jit
        def eval_erg_metric(x, args):
            """ evaluates the ergodic metric on a trjaectory  """
            phik = args['phik']
            e = emap(x, args)
            ck = get_ck(e, self.basis)
            return self.erg_metric(ck, phik)
        self.eval_erg_metric = eval_erg_metric
        
    def get_trajectory(self, max_iter=10000, args=None):
        x = np.linspace(args['x0'], args['xf'], self.time_horizon, endpoint=True)
        u = np.zeros((self.time_horizon, self.robot_model.N, self.robot_model.m))
        self.init_sol = np.concatenate([x, u], axis=2)
        self.solver.set_init_cond(self.init_sol)
        ifConv = self.solver.solve(max_iter=max_iter, args=args)
        sol = self.solver.get_solution()
        x = sol['x'][:,:,:self.robot_model.n]
        u = sol['x'][:,:,self.robot_model.n:]
        self.curr_sol = (x, u)
        return (x, u), ifConv

# <-- example code for how to use 
# if __name__=='__main__':
#     import sys 
#     import matplotlib.pyplot as plt

#     robot_model     = SingleIntegrator()
#     target_distr    = TargetDistribution()
#     basis           = BasisFunc(n_basis=[8,8])
#     args = {
#         'x0' : np.array([2.0,3.25, 0.]),
#         'xf' : np.array([1.75, -0.75, 0.]),
#         'phik' : get_phik(target_distr.evals, basis),
#         'wrksp_bnds' : np.array([[0.,3.5],[-1.,3.5]])
#     }
#     traj_opt = ErgodicTrajectoryOpt(robot_model, basis=basis, time_horizon=250, args=args)
#     print('solving traj')
#     x, u = traj_opt.get_trajectory()

#     # plotting function
#     for obs in traj_opt.obs:
#         _patch = obs.draw()
#         plt.gca().add_patch(_patch)
#     # _mixed_vals = np.inf*np.ones_like(X)
#     # for obs in traj_opt.obs:
#     #     _vals = vmap(obs.distance)(pnts).reshape(X.shape)
#     #     _mixed_vals = np.minimum(_vals, _mixed_vals)
#     #     plt.contour(X, Y, _vals.reshape(X.shape), levels=[-0.01,0.,0.01])
#     plt.plot(x[:,0], x[:,1], 'r')
#     plt.show()