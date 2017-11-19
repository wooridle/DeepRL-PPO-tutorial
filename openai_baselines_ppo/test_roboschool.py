import tf_util as U
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import roboschool
from copy import deepcopy

#from OpenGL import GL

def run():
    import mlp_policy_robo
    U.make_session(num_cpu=1).__enter__()
    env = gym.make("RoboschoolHumanoid-v1")
    #env = wrappers.Monitor(env, directory="./video/HalfCheeta-v1", force=True)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy_robo.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=128, num_hid_layers=2)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)

    U.load_state("save/Humanoid-v1")
    for epi in range(100):
        ob = env.reset()

        total_reward = 0
        step = 0
        while True:
            env.render("human")
            ac, v = pi.act(True, ob)
            
            ob, rew, new, info = env.step(ac)
            step += 1

            total_reward += rew
            
            if new:
                print("Reward: {}, Step: {}".format(total_reward, step))
                break

run()
