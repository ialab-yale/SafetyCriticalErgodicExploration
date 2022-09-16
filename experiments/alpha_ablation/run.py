import sys
sys.path.append('../../')
import numpy as np

from erg_traj_opt_lib.motion_model import SingleIntegrator
from erg_traj_opt_lib.target_distribution import TargetDistribution
from erg_traj_opt_lib.fourier_utils import BasisFunc, get_phik
from erg_traj_opt_lib.erg_traj_opt import ErgodicTrajectoryOpt
from erg_traj_opt_lib.obstacle import Obstacle

import matplotlib.pyplot as plt
from label_lines import *

if __name__=='__main__':


    robot_model     = SingleIntegrator()
    target_distr    = TargetDistribution()
    basis           = BasisFunc(n_basis=[8,8])

    args = {
        'x0' : np.array([-3.0,-0.0]),
        'xf' : np.array([3.0, 0.0]),
        'phik' : get_phik(target_distr.evals, basis),
        'wrksp_bnds' : np.array([[-3.1,3.1],[-2.1,2.1]]),
        'alpha' : 0.99
    }

    obs = [
        Obstacle(pos=np.array([0.,0.]),     half_dims=np.array([0.5,0.5]), th=0.),
        Obstacle(pos=np.array([0.,1.5]),   half_dims=np.array([0.2,0.2]), th=0., p=2),
        Obstacle(pos=np.array([0.,-1.5]),  half_dims=np.array([0.2,0.2]), th=0., p=2),
        Obstacle(pos=np.array([1.5,0.5]),   half_dims=np.array([0.2,0.2]), th=0., p=2),
        Obstacle(pos=np.array([-1.5,0.5]),  half_dims=np.array([0.2,0.2]), th=0., p=2),
        Obstacle(pos=np.array([-1.5,-0.5]), half_dims=np.array([0.2,0.2]), th=0., p=2),
        Obstacle(pos=np.array([1.5,-0.5]),  half_dims=np.array([0.2,0.2]), th=0., p=2),
    ]

    alphas = np.arange(0.1, 1.0, 0.2)
    # alphas = np.array([0.25, 0.50, 0.75, 1.0])

    traj_opt = ErgodicTrajectoryOpt(robot_model, obstacles=obs, basis=basis, time_horizon=200, args=args)

    X, Y = np.meshgrid(*[np.linspace(wks[0],wks[1]) for wks in args['wrksp_bnds']])
    pnts = np.vstack([X.ravel(), Y.ravel()]).T

    _mixed_vals = -np.inf * np.ones_like(X)
    for ob in obs:
        _vals = np.array([ob.distance(pnt) for pnt in pnts]).reshape(X.shape)
        _mixed_vals = np.maximum(_vals, _mixed_vals)
    

    erg_vals = []

    plt.figure(figsize=(3,2))

    for alpha in alphas:

        print('solving traj')
        args.update({'alpha' : alpha})
        (x, u), isConv = traj_opt.get_trajectory(args=args)
        erg_vals.append(traj_opt.eval_erg_metric(x, args))
        # plotting function
        # for obs in traj_opt.obs:
        #     _patch = obs.draw()
        #     plt.gca().add_patch(_patch)
        plt.contour(X, Y, _mixed_vals, levels=[-0.01,0.,0.01], linewidths=2, colors='k')
        # plt.plot(x[:,0], x[:,1], label="{:.1f}".format(alpha), linestyle='dashdot')#, c='m', alpha=alpha)
        plt.plot(x[:,0], x[:,1], linestyle='dashdot')#, c='m', alpha=alpha)
        plt.legend(['0.25, 0.50, 0.75, 1.00'])

        labelLines(plt.gca().get_lines(), align=False, zorder=2.5)
        plt.tight_layout()
        plt.axis('equal')
    
    plt.figure()
    plt.plot(alphas, erg_vals)
    plt.show()