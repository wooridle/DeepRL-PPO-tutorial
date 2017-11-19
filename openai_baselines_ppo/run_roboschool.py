#!/usr/bin/env python
import tf_util as U
import os.path as osp
import gym, logging
import roboschool
import logger
import sys

#from OpenGL import GL

def train(env_id, num_timesteps, seed):
    import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1, num_gpu=0).__enter__()
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=128, num_hid_layers=2)
    env.seed(seed)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=1e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='constant',
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='RoboschoolHumanoid-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.env, num_timesteps=2e7, seed=args.seed)


if __name__ == '__main__':
    main()

