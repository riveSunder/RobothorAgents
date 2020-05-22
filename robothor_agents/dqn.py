#import robothor_challenge
from robothor_agents.env import RobothorChallengeEnv
from robothor_agents.agent import SimpleRandomAgent
from robothor_agents.agent import MyTempAgent, OffTaskModel

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import time
import os

import matplotlib.pyplot as plt


ALLOWED_ACTIONS =  ['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Stop']
 

class DQN():

    def __init__(self, env, agent_fn):

        self.env = env

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
        self.lr = 1e-3
        self.batch_size = 64
        self.buffer_size = 4096 #4096
        self.epsilon_decay = 0.95
        self.starting_epsilon = 0.99
        self.epsilon = self.starting_epsilon * 1.0
        self.min_epsilon = 0.05
        self.update_qt_every = 4
        self.gamma = 0.95

        # flags for gathering off-task training frames
        self.get_depth_frame = True
        self.get_class_frame = True
        self.get_object_frame = False

        # train on this device
        if (1):

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

    def pre_process(self, x):
        # for now just scale the images down to a friendlier size
        # 
        x = F.avg_pool2d(x, 2)

        return x

    def get_episodes(self, steps=128):

        # define lists for recording trajectory transitions
        l_obs_x = torch.Tensor()
        l_obs_one_hot = torch.Tensor()
        l_rew = torch.Tensor()
        l_act = torch.Tensor()
        l_next_obs_x = torch.Tensor()
        l_next_obs_one_hot = torch.Tensor()
        l_done = torch.Tensor()

        # these are optional observables which are to be used for off-task training
        l_depth_frame = torch.Tensor()
        l_class_frame = torch.Tensor()
        l_object_frame = torch.Tensor()
        l_class_labels = torch.Tensor()

        # interaction loop
        done = True
        with torch.no_grad():
            for step in range(steps):

                if done:
                    self.q.reset()


                    observation, info = env.reset(get_depth_frame=self.get_depth_frame,\
                            get_class_frame=self.get_class_frame,\
                            get_object_frame=self.get_object_frame)

                    target_str = observation["object_goal"]

                    target_one_hot = torch.Tensor(\
                            np.array([1.0 if target_str in elem else 0.0 \
                            for elem in self.q.possible_targets]))\
                            .reshape(1,12)

                    obs = [torch.Tensor(observation["rgb"].copy())\
                            .unsqueeze(0).permute(0,3,1,2),\
                            target_one_hot]
                    obs[0] = self.pre_process(obs[0])
                    done = False

                if torch.rand(1) < self.epsilon:
                    act = np.random.randint(len(ALLOWED_ACTIONS)-1)
                    action = ALLOWED_ACTIONS[act]
                    act = torch.Tensor(np.array(1.0*act)).unsqueeze(0)
                else:
                    try:
                        q_values = self.q.forward(obs[0], obs[1])
                    except:
                        import pdb; pdb.set_trace()
                    action = ALLOWED_ACTIONS[torch.argmax(q_values)]
                    act = 1.0*torch.argmax(q_values).unsqueeze(0)


                
                # check to see if target object is nearby, if so, stop (teacher signal)
                if info["target_nearby"]:
                    action = "Stop"
                elif info["advice"] is not None:
                    # get advice from environment if we are close to target
                    # but don't blindly follow advice. Follow it randomly.
                    if np.random.random() < min([self.epsilon*3, 0.5]):
                        action = info["advice"]



                prev_obs = obs
                observation, reward, done, info = self.env.step(action)
                target_str = observation["object_goal"]

                target_one_hot = torch.Tensor(\
                        np.array([1.0 if target_str in elem else 0.0 \
                        for elem in self.q.possible_targets]))\
                        .reshape(1,12)


                obs = [torch.Tensor(observation["rgb"].copy())\
                        .unsqueeze(0).permute(0,3,1,2),\
                        target_one_hot]

                if len(prev_obs[0].shape) == 3:
                    print("dimensional problem?")
                    import pdb; pdb.set_trace()

                obs[0] = self.pre_process(obs[0])



                l_obs_x = torch.cat([l_obs_x, prev_obs[0]], dim=0)
                l_obs_one_hot = torch.cat([l_obs_one_hot, prev_obs[1]], dim=0)
                l_rew = torch.cat([l_rew, torch.Tensor(np.array(1.*reward)).unsqueeze(0)], dim=0)
                l_act = torch.cat([l_act, act], dim=0)
                l_next_obs_x = torch.cat([l_next_obs_x, obs[0]], dim=0)
                l_next_obs_one_hot = torch.cat([l_next_obs_one_hot, obs[1]], dim=0)
                l_done = torch.cat([l_done,torch.Tensor(np.array(1.0*done)).unsqueeze(0)], dim=0)


                if self.get_depth_frame:
                    temp = self.pre_process(torch.Tensor(observation["depth"].copy()).unsqueeze(0))

                    l_depth_frame = torch.cat([l_depth_frame, temp], dim=0)

                if self.get_class_frame:

                    temp = self.pre_process(torch.Tensor(observation["class_frame"]\
                            .copy()).unsqueeze(0).permute(0,3,1,2))

                    l_class_frame = torch.cat([l_class_frame, temp], dim=0)

                if self.get_object_frame:
                    temp = self.pre_process(torch.Tensor(observation["object_frame"]\
                            .copy()).unsqueeze(0))
                    l_object_frame = torch.cat([l_object_frame, temp], dim=0)

                l_class_labels = torch.cat([l_class_labels,\
                        torch.Tensor(observation["class_labels"]).unsqueeze(0)], dim=0)


            print("number of episodes {} with max/mean reward {:.2f}/{:.2f}"\
                    .format(torch.sum(l_done), torch.max(l_rew), torch.sum(l_rew)/torch.sum(l_done)))
            return l_obs_x, l_obs_one_hot, l_rew, l_act,\
                    l_next_obs_x, l_next_obs_one_hot, l_done, \
                    l_class_labels, l_depth_frame, l_class_frame, l_object_frame

    def compute_q_loss(self, t_obs_x, t_obs_one_hot, t_rew, t_act,\
            t_next_obs_x, t_next_obs_one_hot, t_done, double_dqn=True):

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



    def train(self, max_epochs=1024):


        optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

        self.rewards = []
        self.losses = []

        exp_id = str(time.time())[:-3]
        smooth_loss = 10.0

        t0 = time.time()

        for epoch in range(max_epochs):

            t1 = time.time()
            self.q.zero_grad()

            l_obs_x, l_obs_one_hot, l_rew, l_act, l_next_obs_x, \
                    l_next_obs_one_hot, l_done, \
                    l_class_labels, l_depth_frame, \
                    l_class_frame, l_object_frame = self.get_episodes(steps=self.buffer_size)


            self.rewards.append(np.mean(l_rew.detach().numpy()))


            #save trajectories
            timestamp = str(time.time())[:-3]

            my_save = {"l_obs_x": l_obs_x,\
                    "l_obs_one_hot": l_obs_one_hot,\
                    "l_rew": l_rew,\
                    "l_act": l_act,\
                    "l_next_obs_x": l_next_obs_x,\
                    "l_next_obs_one_hot": l_next_obs_one_hot,\
                    "l_done": l_done}
            
            my_save["l_depth_frame"] = l_depth_frame
            my_save["l_class_frame"] = l_class_frame.to(torch.int8)
            my_save["l_object_frame"] = l_object_frame
            my_save["l_class_labels"] = l_class_labels

            torch.save(my_save, "./data/trajectories_{}_{}.pt".format(\
                    self.buffer_size, timestamp))
            
            dir_list = os.listdir("./data")

            # train on all previous experience each time
            t2 = time.time()
            for my_file in dir_list:

                dataset = torch.load("./data/"+my_file)
                l_obs_x = dataset["l_obs_x"]
                l_obs_one_hot = dataset["l_obs_one_hot"]
                l_rew = dataset["l_rew"]
                l_act = dataset["l_act"]
                l_next_obs_x = dataset["l_next_obs_x"]
                l_next_obs_one_hot = dataset["l_next_obs_one_hot"]
                l_done = dataset["l_done"]


                for batch in range(0,self.buffer_size-self.batch_size, self.batch_size):

                    loss = self.compute_q_loss(l_obs_x[batch:batch+self.batch_size],\
                            l_obs_one_hot[batch:batch+self.batch_size], \
                            l_rew[batch:batch+self.batch_size], \
                            l_act[batch:batch+self.batch_size],\
                            l_next_obs_x[batch:batch+self.batch_size], \
                            l_next_obs_one_hot[batch:batch+self.batch_size], \
                            l_done[batch:batch+self.batch_size])

                    loss.backward()
                    smooth_loss = 0.9 * smooth_loss + 0.1 * loss.detach().numpy()

                    optimizer.step()

            self.losses.append(smooth_loss)
            t3 = time.time()
            print("timing simulator: {:.1f} learning: {:.1f} total: {:.1f} s"\
                    .format(t2-t1, t3-t2, t3-t0))
            print("loss at epoch {}: {:.3e} epsilon {:.2e}"\
                    .format(epoch, loss, self.epsilon))

            # update target network every once in a while
            if epoch % self.update_qt_every == 0:

                self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))

                for param in self.qt.parameters():
                    param.requires_grad = False
            
            
            np.save("./logs/losses_{}.npy".format(exp_id), self.losses)
            np.save("./logs/rewards_{}.npy".format(exp_id), self.rewards)

            for my_buffer in [l_obs_x, l_obs_one_hot, l_rew, l_act, l_next_obs_x, \
                    l_next_obs_one_hot,\
                    l_done]:
                del my_buffer



            self.epsilon = np.max([self.min_epsilon, self.epsilon*self.epsilon_decay])
        
if __name__ == "__main__":
    agent_fn = MyTempAgent
    agent = MyTempAgent()

    pretrained = OffTaskModel()
    pretrained.load_state_dict(torch.load("temp_off_task_model.pt"))
    agent.feature_extractor.load_state_dict(pretrained.feature_extractor.state_dict())
    env = RobothorChallengeEnv(agent=agent)
    dqn = DQN(env, agent_fn)


    try:
        dqn.train(max_epochs = 100)
    except KeyboardInterrupt:
        import pdb; pdb.set_trace()

    torch.save(dqn.q.state_dict(),"./my_dqn_temp.pt")
        


