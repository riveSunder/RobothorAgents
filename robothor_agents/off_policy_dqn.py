#import robothor_challenge
from robothor_agents.env import RobothorChallengeEnv
from robothor_agents.agent import SimpleRandomAgent
from robothor_agents.agent import MyTempAgent, OffTaskModel, RandomNetwork

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import time
import os
import argparse
import subprocess
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD

import matplotlib.pyplot as plt


ALLOWED_ACTIONS =  ['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Stop']
 

class DQN():

    def __init__(self, env, agent_fn, use_rnd=False):

        self.env = env
        self.use_rnd = use_rnd

        if self.use_rnd:
            self.rnd_scale = 1e-1
            self.rnd = RandomNetwork()

        self.possible_targets = ['Alarm Clock', \
                            'Apple',\
                            'Baseball Bat',\
                            'Basketball',\
                            'Bowl',\
                            'Garbage Can',\
                            'House Plant',\
                            'Laptop',\
                            'Mug',\
                            'Spray Bottle',\
                            'Television',\
                            'Vase']

        # some training parameters
        self.lr = 1e-5
        self.batch_size = 64
        self.buffer_size = 1024
        self.epsilon_decay = 0.9
        self.starting_epsilon = 0.99
        self.epsilon = self.starting_epsilon * 1.0
        self.min_epsilon = 0.05
        self.update_qt_every = 20
        self.gamma = 0.95

        # flags for gathering off-task training frames
        self.get_depth_frame = True
        self.get_class_frame = True
        self.get_object_frame = False

        # train on this device
        if (0):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        # define the value and value target networks q and q_t

        self.q = agent_fn()
        self.qt = agent_fn()

        self.q.to(self.device)
        self.qt.to(self.device)

        for param in self.qt.parameters():
            param.requires_grad = False

    def pre_process(self, x, norm_max = None):
        # for now just scale the images down to a friendlier size
        # 
        x = F.avg_pool2d(x, 2)

        if norm_max is not None:
            x /= norm_max

        return x

    def get_rnd_reward(self, obs):

        with torch.enable_grad():

            obs = obs.flatten(start_dim=1)
            obs.requires_grad = True

            rnd_reward = self.rnd.get_rnd_reward(obs[0])


        return rnd_reward.detach()


    def compute_q_loss(self, t_obs_x, t_obs_one_hot, t_rew, t_act,\
            t_next_obs_x, t_next_obs_one_hot, t_done, t_info=None, double_dqn=True):

        if t_info is not None:

            for ii in range(len(t_info)):

                if "objects" not in t_info[ii].keys():
                    #import pdb; pdb.set_trace()
                    pass

                else:
                    for my_object in t_info[ii]["objects"]:

                        # below is to avoid conflating e.g. "Baseball" with "BaseballBat"
                        object_name = my_object["name"].split("_")[0].lower()

                        if object_name in [elem.lower() for elem in self.possible_targets]:

                            if my_object["visible"] and my_object["distance"] <= 1.0:

                                # with some probability set target to target in frame if possible
                                my_prob = 0.75
                                if np.random.random() < my_prob:

                                    #import pdb; pdb.set_trace()
                                    target_str = object_name
                                    target_one_hot = torch.Tensor(\
                                            np.array([1.0 if "".join(target_str.split()).lower() in "".join(elem.split()).lower() else 0.0 \
                                            for elem in self.q.possible_targets]))\

                                    t_obs_one_hot[ii] = target_one_hot
                                    t_act[ii] = 6. 
                                    t_done[ii] = 1.0
                                    t_rew[ii] = 10.0



        with torch.no_grad():
            qt = self.qt.forward(t_next_obs_x, t_next_obs_one_hot)
            
            if double_dqn:
                qtq = self.q.forward(t_next_obs_x, t_next_obs_one_hot)
                qt_max = torch.gather(qt, -1, torch.argmax(qtq, dim=-1).unsqueeze(-1))
            else:
                qt_max = torch.gather(qt, -1, torch.argmax(qt, dim=-1).unsqueeze(-1))

            yj = t_rew + ((1-t_done) * self.gamma * qt_max)


        t_act = t_act.long().unsqueeze(1)
        q_av = self.q.forward(t_obs_x, t_obs_one_hot)

        q_act = torch.gather(q_av, -1, t_act)

        loss = torch.mean(torch.pow(yj - q_act, 2))

        return loss



    def train(self, max_epochs=1024, rollouts=False, train_policy=True):


        optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

        self.rewards = []
        self.losses = []

        exp_id = str(time.time())[:-3]
        smooth_loss = 10.0

        t0 = time.time()

        for epoch in range(max_epochs):

            t1 = time.time()
            if rollouts:
                self.q.zero_grad()

                l_obs_x, l_obs_one_hot, l_rew, l_act, l_next_obs_x, \
                        l_next_obs_one_hot, l_done, \
                        l_class_labels, l_depth_frame, \
                        l_class_frame, l_object_frame, l_info = self.get_episodes(steps=self.buffer_size)


                self.rewards.append(np.mean(l_rew.detach().numpy()))


                #save trajectories
                timestamp = str(time.time())[:-3]

                my_save = {"l_obs_x": l_obs_x,\
                        "l_obs_one_hot": l_obs_one_hot,\
                        "l_rew": l_rew,\
                        "l_act": l_act,\
                        "l_next_obs_x": l_next_obs_x,\
                        "l_next_obs_one_hot": l_next_obs_one_hot,\
                        "l_done": l_done,\
                        "l_info": l_info}
                
                my_save["l_depth_frame"] = l_depth_frame
                my_save["l_class_frame"] = l_class_frame.to(torch.int8)
                my_save["l_object_frame"] = l_object_frame
                my_save["l_class_labels"] = l_class_labels

                torch.save(my_save, "./data/trajectories_winfo_{}_{}.pt".format(\
                        self.buffer_size, timestamp))
                
                t2 = time.time()

                for my_buffer in [l_obs_x, l_obs_one_hot, l_rew, l_act, l_next_obs_x, \
                        l_next_obs_one_hot,\
                        l_done]:
                    del my_buffer

            if train_policy:


                dir_list = os.listdir("./data")
                len_dir_list = len(dir_list)
                num_samples = 30

                # train on a sample of all previous experience each time
                t2 = time.time()

                self.q.to(self.device)
                self.qt.to(self.device)

                for sample in range(num_samples):
                    my_file = dir_list[np.random.randint(len_dir_list)]

                    dataset = torch.load("./data/"+my_file)
                    l_obs_x = dataset["l_obs_x"]
                    l_obs_one_hot = dataset["l_obs_one_hot"]
                    l_rew = dataset["l_rew"]
                    l_act = dataset["l_act"]
                    l_next_obs_x = dataset["l_next_obs_x"]
                    l_next_obs_one_hot = dataset["l_next_obs_one_hot"]
                    l_done = dataset["l_done"]
                    l_info = dataset["l_info"]

                    # send to device
                    l_obs_x = l_obs_x.to(self.device)
                    l_obs_one_hot = l_obs_one_hot.to(self.device)
                    l_rew = l_rew.to(self.device)
                    l_act = l_act.to(self.device)
                    l_next_obs_x = l_next_obs_x.to(self.device)
                    l_next_obs_one_hot = l_next_obs_one_hot.to(self.device)
                    l_done = l_done.to(self.device)

                    for batch in range(0,self.buffer_size-self.batch_size-1, self.batch_size):

                        loss = self.compute_q_loss(l_obs_x[batch:batch+self.batch_size],\
                                l_obs_one_hot[batch:batch+self.batch_size], \
                                l_rew[batch:batch+self.batch_size], \
                                l_act[batch:batch+self.batch_size],\
                                l_next_obs_x[batch:batch+self.batch_size], \
                                l_next_obs_one_hot[batch:batch+self.batch_size], \
                                l_done[batch:batch+self.batch_size], 
                                l_info[batch:batch+self.batch_size])

                        loss.backward()
                        smooth_loss = 0.9 * smooth_loss + 0.1 * loss.detach().cpu().numpy()

                        optimizer.step()

                self.losses.append(smooth_loss)
                self.epsilon = np.max([self.min_epsilon, self.epsilon*self.epsilon_decay])

                # update target network every once in a while
                if epoch % self.update_qt_every == 0:
                    torch.save(self.q.state_dict(),"./my_dqn_temp.pt")

                    self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))

                    for param in self.qt.parameters():
                        param.requires_grad = False
            
            
                np.save("./logs/losses_{}.npy".format(exp_id), self.losses)
                print("loss at epoch {}: {:.3e} epsilon {:.2e}"\
                        .format(epoch, loss, self.epsilon))

                for my_buffer in [l_obs_x, l_obs_one_hot, l_rew, l_act, l_next_obs_x, \
                        l_next_obs_one_hot,\
                        l_done]:
                    del my_buffer

            t3 = time.time()
            print("timing simulator: {:.1f} learning: {:.1f} total: {:.1f} s"\
                    .format(t2-t1, t3-t2, t3-t0))

            if rollouts and train_policy:
                np.save("./logs/rewards_{}.npy".format(exp_id), self.rewards)


def train():

    agent_fn = MyTempAgent
    agent = MyTempAgent()

    env = None #RobothorChallengeEnv(agent=agent)
    dqn = DQN(env, agent_fn, use_rnd=True)
    if(0):
        pretrained = OffTaskModel()
        pretrained.load_state_dict(torch.load("temp_off_task_model.pt"))
        agent.feature_extractor.load_state_dict(pretrained.feature_extractor.state_dict())
    else:
        dqn.q.load_state_dict(torch.load("./my_dqn_temp.pt"))
        dqn.qt.load_state_dict(torch.load("./my_dqn_temp.pt"))

    t0 = time.time()
    epochs = 1000

    dqn.train(max_epochs=epochs, rollouts=False, train_policy=True)
    torch.save(dqn.q.state_dict(),"./my_dqn_temp.pt")





if __name__ == "__main__":


   train()

        


