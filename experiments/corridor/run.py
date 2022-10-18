import sys
sys.path.append('../../')
import numpy as np

from erg_traj_opt_lib.motion_model import MultiRobotSingleIntegrator
from erg_traj_opt_lib.target_distribution import TargetDistribution
from erg_traj_opt_lib.fourier_utils import BasisFunc, get_phik
from erg_traj_opt_lib.ma_erg_traj_opt import MAErgodicTrajectoryOpt
from erg_traj_opt_lib.obstacle import Obstacle

import matplotlib.pyplot as plt

if __name__=='__main__':


    robot_model     = MultiRobotSingleIntegrator()
    n_states        = robot_model.n
    N_robots        = robot_model.N
    m_ctrls         = robot_model.m

    target_distr    = TargetDistribution()
    basis           = BasisFunc(n_basis=[8,8])

    time_horizon = 100
    
    np.random.seed(10)  # used for random drone positioning

    x0 = np.array([
        [0.25, 0.25],
        [1.75, 1.75]
    ])
    
    xf = np.array([
        [1.75, 1.75],
        [0.25, 0.25]
    ])

    args = {
        # 'x0' : -np.ones((N_robots, n_states))+np.random.normal(0., 0.1, size=(N_robots, n_states)),
        # 'xf' :  np.ones((N_robots, n_states))+np.random.normal(0., 0.3, size=(N_robots, n_states)),
        'x0' : x0,
        'xf' : xf,
        'phik' : get_phik(target_distr.evals, basis),
        'wrksp_bnds' : np.array([[0,2.0],[0,2.0]]),
        'alpha' : 0.2
    }

    # Set up corridor
    R_drone = 0.07      # radius of drone in m
    w = 1.0 - R_drone   # width of left/right sides of the corridore for a 2.0 x 2.0 workspace
    h = 1.0             # height of the corridor, gives each drone a 1.0 x 0.5 space to search in
    obs = [
        Obstacle(pos=np.array([w/2, 1.0]),             half_dims=np.array([w/2,h/2]), th=0., buff=0.0, p=12),
        Obstacle(pos=np.array([3*w/2 + R_drone, 1.0]), half_dims=np.array([w/2,h/2]), th=0., buff=0.0, p=12),
    ]

    traj_opt = MAErgodicTrajectoryOpt(robot_model, obstacles=obs, basis=basis, time_horizon=200, args=args)

    X, Y = np.meshgrid(*[np.linspace(wks[0],wks[1]) for wks in args['wrksp_bnds']])
    pnts = np.vstack([X.ravel(), Y.ravel()]).T

    _mixed_vals = -np.inf * np.ones_like(X)
    for ob in obs:
        _vals = np.array([ob.distance(pnt) for pnt in pnts]).reshape(X.shape)
        _mixed_vals = np.maximum(_vals, _mixed_vals)
    
    # plt.figure(figsize=(3,2))

    print('solving traj')
    (x, u), isConv = traj_opt.get_trajectory(args=args)

    plt.contour(X, Y, _mixed_vals, levels=[-0.01,0.,0.01], linewidths=2, colors='k')
    for i in range(robot_model.N):
        plt.plot(x[:,i, 0], x[:,i, 1], linestyle='dashdot')#, c='m', alpha=alpha)
    # plt.tight_layout()
    # plt.axis('equal')

    plt.figure()
    plt.plot(x[:,0,:2]-x[:,1,:2])
    # for i in range(robot_model.N):
    #     plt.plot(x[:,i,:2])    

    plt.show()