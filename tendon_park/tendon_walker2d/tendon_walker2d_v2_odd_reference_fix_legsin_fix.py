import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
import pdb
import os

class TendonWalker2dv2OddReferenceEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, cycle_of_ref, ref_coeff):
        self.time_step = 0
        self.timestep_limit = 2000
        self.cycle_of_reference = cycle_of_ref
        self.ref_coeff = ref_coeff
        self.reference_motion_right = (np.sin(np.arange(self.timestep_limit)*(math.pi * 2./self.cycle_of_reference)) + 1.)
        self.reference_motion_right_leg = - (np.sin(np.arange(self.timestep_limit)*(math.pi * 2./self.cycle_of_reference) + math.pi) + 1.)
        self.reference_motion_left = np.zeros(self.timestep_limit)
        self.reference_motion_left_leg = - (np.sin(np.arange(self.timestep_limit)*(math.pi * 2./self.cycle_of_reference) + 1.5 * math.pi + math.pi) + 1.)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.getcwd(), "tendon_walker_v2_fix.xml"), 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        #posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.
        reward = alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward -= self.ref_coeff * ((self.reference_motion_right[self.time_step] - self.sim.data.qpos[0]) ** 2 + (self.reference_motion_left[self.time_step] - self.sim.data.qpos[3]) ** 2 +(self.reference_motion_right_leg[self.time_step] - self.sim.data.qpos[1]) ** 2 + (self.reference_motion_left_leg[self.time_step] - self.sim.data.qpos[4]) ** 2)
        #print("ref {} and qpos {}".format(self.reference_motion_right[self.time_step],self.sim.data.qpos[0]))
        done = False#not (height > 0.8 and height < 2. and ang > -1.5 and ang < 1.5)
        ob = self._get_obs()
        self.time_step += 1

        #pdb.set_trace()
        return ob, reward, done, {}
    
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nv)
        )
        self.time_step = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
