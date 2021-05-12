import torch
import numpy as np
from collections import deque
from dqn_agent import Agent
from unityagents import UnityEnvironment


def dqn(env, agent, brain_name, n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]  
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break 
                
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window)}", end="")
        
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window)}")
            
        if np.mean(scores_window) >= 13.0:
            print(f"\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window)}")
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break
            
    return scores


def train(unity_environment: str="./Banana_Linux/Banana.x86_64", seed: int=42):

    env = UnityEnvironment(file_name=unity_environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    agent = Agent(
        state_size=len(env_info.vector_observations[0]), 
        action_size=brain.vector_action_space_size, 
        seed=seed
        )

    scores = dqn(env, agent, brain_name)

    env.close()

    return scores
