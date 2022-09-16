from parser import sequence2st
import time
import numpy as np

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm


# URIs for the swarm, change the last digit to match the crazyflies you are using.
URI0 = 'radio://0/80/2M/E7E7E7E700'
URI1 = 'radio://0/80/2M/E7E7E7E701'
URI2 = 'radio://0/80/2M/E7E7E7E702'
URI3 = 'radio://0/80/2M/E7E7E7E703'

z0 = 0.4
dt = 0.1

sequence0 = []
sequence1 = []
sequence2 = []
sequence3 = []

for i in np.arange(4):
    if i == 0:
        sol  = np.load('optimized_trajectories_0.npy', allow_pickle=False)
        for j in range(sol.shape[1]):
            sequence0.append([sol[0,j], sol[1,j], z0, dt])
    elif i==1:
        sol  = np.load('optimized_trajectories_1.npy', allow_pickle=False)
        for j in range(sol.shape[1]):
            sequence1.append([sol[0,j], sol[1,j], z0, dt])
    elif i==2:
        sol  = np.load('optimized_trajectories_2.npy', allow_pickle=False)
        for j in range(sol.shape[1]):
            sequence2.append([sol[0,j], sol[1,j], z0, dt])
    elif i==3:
        sol  = np.load('optimized_trajectories_3.npy', allow_pickle=False)
        for j in range(sol.shape[1]):
            sequence3.append([sol[0,j], sol[1,j], z0, dt])

seq_args = {
    URI0: [sequence0],
    URI1: [sequence1],
    URI2: [sequence2],
    URI3: [sequence3],
}

# List of URIs, comment the one you do not want to fly
uris = {
    URI0,
    URI1,
    URI2,
    URI3,
}

def wait_for_param_download(scf):
    while not scf.cf.param.is_updated:
        time.sleep(1.0)
    print('Parameters downloaded for', scf.cf.link_uri)


def take_off(cf, position):
    take_off_time = 1.5
    sleep_time = 0.1
    steps = int(take_off_time / sleep_time)
    vz = position[2] / take_off_time

    for i in range(steps):
        cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
        time.sleep(sleep_time)

def land(cf, position):
    landing_time = 0.5
    sleep_time = 0.1
    steps = int(landing_time / sleep_time)
    vz = -position[2] / landing_time

    for _ in range(steps):
        cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
        time.sleep(sleep_time)

    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)

def run_sequence(scf, sequence):
    try:
        cf = scf.cf

        take_off(cf, sequence[0])
        for position in sequence:
            print('Setting position {}'.format(position))
            end_time = time.time() + position[3]
            while time.time() < end_time:
                cf.commander.send_position_setpoint(position[0],
                                                    position[1],
                                                    position[2], 0)
                time.sleep(0.1)
        land(cf, sequence[-1])
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    cflib.crtp.init_drivers()

    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        # If the copters are started in their correct positions this is
        # probably not needed. The Kalman filter will have time to converge
        # any way since it takes a while to start them all up and connect. We
        # keep the code here to illustrate how to do it.
        # swarm.reset_estimators()

        # The current values of all parameters are downloaded as a part of the
        # connections sequence. Since we have 10 copters this is clogging up
        # communication and we have to wait for it to finish before we start
        # flying.
        print('Waiting for parameters to be downloaded...')
        swarm.parallel(wait_for_param_download)

        swarm.parallel(run_sequence, args_dict=seq_args)
