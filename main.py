#!/usr/bin/env python3 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname((os.path.abspath(os.path.dirname(__file__)))))))


import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import csv
import pickle

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from environments.DiffDriveEnv import *
from environments.DiffDriveSensorEnv import *
from environments.GaitDiscreteEnv import *

# def save_learning_curve(output, episode_rewards):
#     with open(output + '_learning_curve.csv', 'w', newline='') as csvfile:
#         fieldnames = ['Episode', 'AverageReward']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         writer.writeheader()
#         for episode, reward in enumerate(episode_rewards):
#             writer.writerow({'Episode': episode, 'AverageReward': reward})

def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    avg_episode_rewards = []
    last_states = []

    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset()[0] if type(env.reset()) is tuple else env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, truncated, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))
            avg_episode_rewards.append(episode_reward)
            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )
            last_states.append(observation) # save last states
            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
        
    last_states_complete = np.array(last_states)
    with open('{}/last_states.pkl'.format(output), 'wb') as f:
        pickle.dump(last_states_complete, f)


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='Pendulum-v1', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=1000, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=500000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--model_path', type=str, help="Model path to load for testing")
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    
    
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    # env = NormalizedEnv(gym.make(args.env))
    
    if args.env == "Diffcar" or args.env == "DiffcarSensor" or args.env == "GaitDiscrete":   
        obsts = Obstacles(fname = "obstacles.mat")

        if args.env == "Diffcar":
            args.output = get_output_folder(args.output, 'Diffcar')
            env = DiffDriveEnv(obstacles=obsts, render_mode='human',
                            obs_space_size = np.array([0,50,0,50]), 
                            init_pos= [5,5,np.pi/2], target_pos = [45,45,np.pi/4],
                            reward_weight=[1.0, 100.0, 0.0, 500.0, 100.0], suc_tol = 2, hpfname = "vecfield.mat")        
        elif args.env == "GaitDiscrete":
            args.output = get_output_folder(args.output, 'GaitDiscrete')    
            env = GaitDiscreteEnv("gait_data.mat",obstacles=obsts, render_mode='human',
                                obs_space_size = np.array([0,50,0,50]), 
                                init_pos= [5,5,0], target_pos = [45,45,np.pi/4],
                                reward_weight=[1.0, 10.0, 10.0, 100.0, 100.0], suc_tol = 2, hpfname = "vecfield.mat")
        elif args.env == "DiffcarSensor":            
            args.output = get_output_folder(args.output, 'DiffcarSensor')    
            env = DiffDriveSensorEnv(obstacles=obsts, render_mode='human',
                                obs_space_size = np.array([0,50,0,50]), 
                                init_pos= [5,5,0], target_pos = [45,45,np.pi/4],
                                reward_weight=[1.0, 100.0, 10.0, 100.0, 100.0], suc_tol = 2, hpfname = "vecfield.mat")
        env.reset()
        env.plot(True)
    else:
        args.output = get_output_folder(args.output, args.env)
        env = gym.make(args.env, render_mode='human')

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    if args.env == "Diffcar" or args.env == "GaitDiscrete":
        nb_states = env.observation_space['agent'].shape[0]
    elif args.env == "DiffcarSensor":
        nb_states = env.observation_space['agent'].shape[0]+env.observation_space['potential'].shape[0]*env.observation_space['potential'].shape[1]
    else:
        nb_states = env.observation_space.shape[0]

    nb_actions = env.action_space.shape[0]


    agent = DDPG(nb_states, nb_actions, env, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.model_path,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))


