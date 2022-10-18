### Jax/numpoy imports 
from mimetypes import init
import jax
import jax.numpy as np

from functools import partial
from jax import value_and_grad, jacfwd, vmap, jit, hessian
from jax.lax import scan
import jax.random as jnp_random
from jax.flatten_util import ravel_pytree
import numpy as onp

### TODO
import pickle as pkl ## <--- this will probably get pushed up to user side

### Local imports
from .solver import AugmentedLagrangian
from .motion_model import SingleIntegrator
from .ergodic_metric import ErgodicMetric
from .obstacle import Obstacle
from .cbf_utils import sdf2cbf
from .fourier_utils import BasisFunc, get_phik, get_ck
from .target_distribution import TargetDistribution

def motion_model_wrapper(f, f_erg, _emap):
    def _f(z, u, args):
        x1, c1 = z
        x2 = f(x1, u)
        c2 = f_erg(_emap(x1, args), c1, args['phik'])
        return (x2, c2), np.concatenate([x2,c2])
    return _f
    
# def motion_model_wrapper(f):
#     def _f(x1, u):
#         x2 = f(x1, u)
#         return x2, x2
#     return _f

class MPCSolver(object):
    def __init__(self, init_sol, loss) -> None:
        '''WARNING THIS LIBRARY SUCKS A LOT'''
        self.solution = init_sol
        self._flat_solution, self._unravel = ravel_pytree(self.solution)
        # self.loss = lambda fl_x, args: loss(self._unravel(fl_x), args)
        self.loss = loss
        self.val_dldu = jit(value_and_grad(self.loss))
        self.dl2du2 = jit(hessian(self.loss))
    def solver_step(self, args, step_size=1e-1, max_iter=1000, eps=1e-6):
        self._shift_solution()
        _eps = 1.0
        _prev_val = None
        for k in range(max_iter):
            # _val, _dldu   = self.val_dldu(self._flat_solution, args)
            # _dl2du2 = self.dl2du2(self._flat_solution, args)
            # du = np.linalg.solve(_dl2du2, -step_size*_dldu)
            # self._flat_solution = self._flat_solution + du
            _val, _dldu   = self.val_dldu(self.solution, args)
            # for _key in self.solution:
            #     self.solution[_key]   = self.solution[_key] - step_size * _dldu[_key]

            self.solution['u']   = self.solution['u']   - step_size * _dldu['u']
            self.solution['mu']   = self.solution['mu'] - step_size * _dldu['mu']

            if _prev_val is None:
                _prev_val = _val
            else:
                _eps = np.abs(_val-_prev_val)
                _prev_val = _val
            if _eps < eps: 
                print('done in ', k, ' iterations')
                break 
        print('solution obtained in ', k, ' iterations')
        # self.solution = self._unravel(self._flat_solution)
        u_app = np.array(self.solution['u'][0])
        return u_app
        
    def _shift_solution(self):
        u = self.solution['u']
        u.at[:-1,:].set(u[1:,:])
        u.at[-1,:].set(0.)
        # u = index_update(u, index[:-1,:], u[1:,:])
        # u = index_update(u, index[-1,:], 0.)
        self.solution.update({'u': u})

def _emap(x, args):
    """ Function that maps states to workspace """
    wrksp_bnds = args['wrksp_bnds']
    return np.array([
        (x[0]-wrksp_bnds[0,0])/(wrksp_bnds[0,1]-wrksp_bnds[0,0]), 
        (x[1]-wrksp_bnds[1,0])/(wrksp_bnds[1,1]-wrksp_bnds[1,0])])
emap = vmap(_emap, in_axes=(0, None))

class ErgodicMPC(object):
    def __init__(self, robot_model, args, obstacles, 
                        basis=None, time_horizon=500) -> None:
        self.robot_model = robot_model
        if basis is None:
            self.basis = BasisFunc(n_basis=[8]*2)
        else:
            self.basis = basis
        self.erg_metric      = ErgodicMetric(self.basis)
        self.erg_dyn = self.erg_metric.construct_erg_dyn(self.robot_model)
        self.traj_sim = motion_model_wrapper(self.robot_model.f, self.erg_dyn, _emap)
        # self.traj_sim = motion_model_wrapper(self.robot_model.f)

        n,m = self.robot_model.n, self.robot_model.m

        ### initial conditions 
        # obs_info = pkl.load(open('obs_info.pkl', 'rb'))
        # self.obs = []
        # for obs_name in obs_info:
        #     _obs = Obstacle(
        #         pos=np.array(obs_info[obs_name]['pos']), 
        #         half_dims=np.array(obs_info[obs_name]['half_dims']),
        #         th=obs_info[obs_name]['rot']
        #     )
        #     self.obs.append(_obs)
        self.obs = obstacles
        self.cbf_consts = []
        for obs in self.obs: 
            self.cbf_consts.append(sdf2cbf(self.robot_model.f, obs.distance, alpha=0.1))

        def barrier_cost(e):
            """ Barrier function to avoid robot going out of workspace """
            return (np.maximum(0, e-1) + np.maximum(0, -e))**2
        
        def terminal_cost(xf, args):
            return np.sum((xf-args['xf'])**2)
        
        def ineq_constr(x, u, args):
            """ control inequality constraints"""
            # p = x[:,:2] # extract just the position component of the trajectory
            # obs_val = [vmap(_ob.distance)(p).flatten() for _ob in self.obs]
            obs_val = [vmap(_cbf_ineq)(x, u).flatten() for _cbf_ineq in self.cbf_consts]
            # ctrl_box = [(np.abs(u) - 1.0).flatten()]
            # _ineq_list = ctrl_box + obs_val
            _ineq_list = obs_val
            return np.concatenate(_ineq_list)
        @jit
        def loss(z, args):
            """ Traj opt loss function, not the same as erg metric """
            x0 = args['x0']
            c0 = args['c0']
            # x_past = np.stack(args['x_past'])
            c  = args['c']
            u  = z['u']
            mu = z['mu']
            (xf, cf), z        = scan(partial(self.traj_sim, args=args), (x0, c0), u)
            x, ck = z[:,:n], z[:,n:]
            _ineq_constr = ineq_constr(x, u, args)
            phik = args['phik']
            e  = emap(x, args)
            # ck = get_ck(e, self.basis)
            return self.erg_metric.eval_delta_cost(ck) \
                    + 0.01 * np.mean(u**2) \
                    + np.sum(barrier_cost(e)) \
                    # + terminal_cost(xf, args)\
                    # + c*0.5 * np.sum(np.maximum(0., mu*_ineq_constr))
                    # + (1/c)*0.5 * np.sum(np.maximum(0., mu + c*_ineq_constr)**2 - mu**2)
        init_u = np.zeros((time_horizon, m))
        self.init_sol = {
                'u' :init_u,
        }        
        zf, z = scan(partial(self.traj_sim, args=args), (args['x0'], args['c0']), init_u)
        x, ck = z[:,:n], z[:,n:]

        _ineq_constr = ineq_constr(x, init_u, args)
        self.init_sol.update({'mu' : np.ones_like(_ineq_constr)})
        self.curr_sol = x
        self.ineq_constr = ineq_constr
        self.mpc_solver = MPCSolver(self.init_sol, loss)
    
    def calc_ctrl(self, args, max_iter=1000):
        u_app = self.mpc_solver.solver_step(args, max_iter=max_iter)
        # xf, x = scan(self.traj_sim, args['x0'], self.mpc_solver.solution['u'])
        zf, z = scan(partial(self.traj_sim, args=args), (args['x0'], args['c0']), self.mpc_solver.solution['u'])
        x, ck = z[:,:self.robot_model.n], z[:,self.robot_model.n:]
        self.curr_sol = (x, u_app)
        c_new = self.erg_dyn(args['x0'], args['c0'], args['phik'])
        return x, u_app, c_new

if __name__=='__main__':
    import sys 
    import matplotlib.pyplot as plt

    robot_model     = SingleIntegrator()
    target_distr    = TargetDistribution()
    basis           = BasisFunc(n_basis=[8,8])
    args = {
        'x0' : np.array([2.0,3.25]),
        'c0' : np.zeros(basis.k_list.shape[0]),
        'xf' : np.array([1.75, -0.75]),
        'c' : 0.1,
        'phik' : get_phik(target_distr.evals, basis),
        'wrksp_bnds' : np.array([[0.,3.5],[-1.,3.5]])
    }
    erg_mpc = ErgodicMPC(robot_model, basis=basis, time_horizon=200, args=args)
    print('solving traj')
    x, u_app = erg_mpc.calc_ctrl(args)

    # plotting function
    for obs in erg_mpc.obs:
        _patch = obs.draw()
        plt.gca().add_patch(_patch)
    # _mixed_vals = np.inf*np.ones_like(X)
    # for obs in erg_mpc.obs:
    #     _vals = vmap(obs.distance)(pnts).reshape(X.shape)
    #     _mixed_vals = np.minimum(_vals, _mixed_vals)
    #     plt.contour(X, Y, _vals.reshape(X.shape), levels=[-0.01,0.,0.01])
    plt.plot(x[:,0], x[:,1], 'r')
    plt.show()