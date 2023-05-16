import sys
sys.path.append('../../')
import numpy as np
import time

from erg_traj_opt_lib.motion_model import MultiRobotSingleIntegrator
from erg_traj_opt_lib.target_distribution import TargetDistribution
from erg_traj_opt_lib.fourier_utils import BasisFunc, get_phik
from erg_traj_opt_lib.ma_erg_traj_opt import MAErgodicTrajectoryOpt
from erg_traj_opt_lib.obstacle import Obstacle

import matplotlib.pyplot as plt
import pickle as pkl

if __name__=='__main__':


    robot_model     = MultiRobotSingleIntegrator(N=4)
    n_states        = robot_model.n
    N_robots        = robot_model.N
    m_ctrls         = robot_model.m

    target_distr    = TargetDistribution()
    basis           = BasisFunc(n_basis=[8,8])

    time_horizon = 100
    np.random.seed(10)

    x0 = np.array([
        [1.0, -0.5],
        [2.5, -0.5],
        [1.0,  2.5],
        [3.0, 3.0]
    ])
    xf = np.array([
        [1.0, 2.5],
        [3.0, 3.0],
        [1.0, -0.5],
        [2.5, -0.5]
    ])
    args = {
        'x0' : x0,
        'xf' : xf,
        'phik' : get_phik(target_distr.evals, basis),
        'wrksp_bnds' : np.array([[0.,3.5],[-1.,3.5]]),
        'alpha' : 0.1
    }

    obs_info = pkl.load(open('../../erg_traj_opt_lib/obs_info.pkl', 'rb'))
    obs = []
    for obs_name in obs_info:
        _obs = Obstacle(
            pos=np.array(obs_info[obs_name]['pos']), 
            half_dims=np.array(obs_info[obs_name]['half_dims']),
            th=obs_info[obs_name]['rot']
        )
        obs.append(_obs)

    traj_opt = MAErgodicTrajectoryOpt(robot_model, obstacles=obs, basis=basis, time_horizon=200, args=args)

    X, Y = np.meshgrid(*[np.linspace(wks[0],wks[1]) for wks in args['wrksp_bnds']])
    pnts = np.vstack([X.ravel(), Y.ravel()]).T

    _mixed_vals = -np.inf * np.ones_like(X)
    for ob in obs:
        _vals = np.array([ob.distance(pnt) for pnt in pnts]).reshape(X.shape)
        _mixed_vals = np.maximum(_vals, _mixed_vals)
    

    # plt.figure(figsize=(3,2))

    max_N    = 10
    succ_cnt = 0
    min_dist = 0.5
    inits = np.array([])
    finals = np.array([])

    print('solving traj')
    start = time.time()

    while succ_cnt < max_N:
        (x, u), isConv = traj_opt.get_trajectory(args=args, max_iter=20000)

        _ineq = traj_opt.ineq_constr(traj_opt.sol['x'], args)

        succ_cnt += 1
    
    end = time.time()
    delta = end - start

    plt.figure(2)
    plt.plot(_ineq)
    plt.show()
    plt.contour(X, Y, _mixed_vals, levels=[-0.01,0.,0.01], linewidths=2, colors='k')
    for i in range(robot_model.N):
        plt.plot(x[:,i, 0], x[:,i, 1], linestyle='dashdot')#, c='m', alpha=alpha)
        # plt.scatter(x[0,i,0], x[0, i, 1], s=80, facecolors='none',)
        plt.scatter(x[-1,i,0], x[-1, i, 1], s=80, marker='o')
        fname = 'optimized_trajectories_' + str(i) + '.npy'
        with open(fname, 'wb') as f:
            np.save(f, np.array([x[:,i, 0], x[:,i, 1]]))
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.legend(['Robot 1', 'Robot 2', 'Robot 3', 'Robot 4'], loc='upper right')
    # axes = plt.gca()
    # axes.set_aspect(0.7777)
    # plt.tight_layout()
    # plt.axis('equal')

    plt.figure()
    plt.plot(x[:,0,:2]-x[:,1,:2])
    # for i in range(robot_model.N):
    #     plt.plot(x[:,i,:2])    

    plt.show()
