from model import ActorNetwork, CriticNetwork
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os


class Actor:

    def __init__(self, device, agent_i, state_size, action_size, random_seed,
        memory, noise,lr, weight_decay,load_params,checkpoint_folder = './'):   

        self.DEVICE = device

        self.agent_i = agent_i

        self.state_size = state_size        
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.LR = lr        
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        self.local = ActorNetwork(state_size, action_size, random_seed).to(self.DEVICE)
        self.target = ActorNetwork(state_size, action_size, random_seed).to(self.DEVICE)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.LR)
        self.checkpoint_full_name = self.CHECKPOINT_FOLDER + 'cp_actor_' + str(self.agent_i) + '.pth'

        self.memory = memory
        self.noise = noise

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.local.eval()
        with torch.no_grad():
            action = self.local(state).cpu().data.numpy()
        self.local.train()        

        if add_noise:
            action += self.noise.sample()
    
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done)

    def reset(self):
        self.noise.reset()
     
    def checkpoint(self):
        torch.save(self.local.state_dict(), self.checkpoint_full_name)
        



class Critic:        

    def __init__(self, device, state_size, action_size, random_seed, gamma, 
                TAU, lr, weight_decay, checkpoint_folder = './'):        

        self.DEVICE = device

        self.state_size = state_size        
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.GAMMA = gamma
        self.TAU = TAU
        self.LR = lr
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder
        
        self.local = CriticNetwork(state_size, action_size, random_seed).to(self.DEVICE)
        self.target = CriticNetwork(state_size, action_size, random_seed).to(self.DEVICE)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        self.checkpoint_full_name = self.CHECKPOINT_FOLDER + 'cp_critic.pth'

    def step(self, actor, memory):
        experiences = memory.sample()        
        if not experiences:
            return
        self.learn(actor, experiences)

    def learn(self, actor, experiences):

  
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = actor.target(next_states)
        Q_targets_next = self.target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.local.parameters(), 1)
        self.optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = actor.local(states)
        actor_loss = - self.local(states, actions_pred).mean()
        # Minimize the loss
        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.local, self.target)
        self.soft_update(actor.local, actor.target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    def checkpoint(self):          
        torch.save(self.local.state_dict(), self.checkpoint_full_name)  