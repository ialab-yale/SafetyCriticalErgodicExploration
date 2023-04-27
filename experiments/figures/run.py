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
        'x0' : np.array([0.25, 0.75]),
        'xf' : np.array([0.75, 0.25]),
        'phik' : get_phik(target_distr.evals, basis),
        'wrksp_bnds' : np.array([[0.0,1.0],[0.0,1.0]]),
        'alpha' : 0.99
    }

    obs = [
        Obstacle(pos=np.array([2.0,2.0]), half_dims=np.array([0.5,0.5]), th=0.),
    ]

    alphas = np.array([0.1])
    # alphas = np.arange(0.1, 1.0, 0.2)
    # alphas = np.array([0.25, 0.50, 0.75, 1.0])

    traj_opt = ErgodicTrajectoryOpt(robot_model, obstacles=obs, basis=basis, time_horizon=100, args=args)

    X, Y = np.meshgrid(*[np.linspace(wks[0],wks[1]) for wks in args['wrksp_bnds']])
    pnts = np.vstack([X.ravel(), Y.ravel()]).T

    _mixed_vals = -np.inf * np.ones_like(X)
    for ob in obs:
        _vals = np.array([ob.distance(pnt) for pnt in pnts]).reshape(X.shape)
        _mixed_vals = np.maximum(_vals, _mixed_vals)
    
    (x, u), isConv = traj_opt.get_trajectory(args=args)
    
    tar_X, tar_Y = target_distr.domain
    _s           = target_distr._s
    vals         = target_distr.evals[0]

    plt.figure()
    plt.contour(tar_X, tar_Y, vals.reshape(tar_X.shape), linewidths=2, colors='k')
    plt.plot(x[:,0], x[:, 1], linestyle='dashdot')#, c='m', alpha=alpha)
    plt.scatter(x[-1,0], x[-1, 1], s=80, marker='o', color='r')
    plt.scatter(x[0,0], x[0, 1], s=80, marker='o', color='g')
    plt.savefig('trajectory.png')