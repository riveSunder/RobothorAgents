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
            self.rnd_scale = 2.0
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
        self.lr = 3e-4
        self.batch_size = 64
        self.buffer_size = 1024
        self.epsilon_decay = 0.95
        self.starting_epsilon = 0.25
        self.epsilon = self.starting_epsilon * 1.0
        self.min_epsilon = 0.05
        self.update_qt_every = 10
        self.gamma = 0.975

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
        l_info = []
        with torch.no_grad():
            for step in range(steps):

                if done:
                    self.q.reset()


                    observation, info = self.env.reset(get_depth_frame=self.get_depth_frame,\
                            get_class_frame=self.get_class_frame,\
                            get_object_frame=self.get_object_frame)

                    target_str = observation["object_goal"]

                    target_one_hot = torch.Tensor(\
                            np.array([1.0 if "".join(target_str.split()).lower() in "".join(elem.split()).lower() else 0.0 \
                            for elem in self.q.possible_targets]))\
                            .reshape(1,12)

                    obs = [torch.Tensor(observation["rgb"].copy())\
                            .unsqueeze(0).permute(0,3,1,2),\
                            target_one_hot]
                    obs[0] = self.pre_process(obs[0], 255.0)
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

                # remember info (including metadata) so we can 
                # implement hindsight experience replay during off-policy training
                l_info.append(info)

                prev_obs = obs
                observation, reward, done, info = self.env.step(action)

                target_str = observation["object_goal"]

                target_one_hot = torch.Tensor(\
                        np.array([1.0 if "".join(target_str.split()).lower() in "".join(elem.split()).lower() else 0.0 \
                        for elem in self.q.possible_targets]))\
                        .reshape(1,12)


                obs = [torch.Tensor(observation["rgb"].copy())\
                        .unsqueeze(0).permute(0,3,1,2),\
                        target_one_hot]

                if len(prev_obs[0].shape) == 3:
                    print("dimensional problem?")
                    import pdb; pdb.set_trace()

                obs[0] = self.pre_process(obs[0], 255.0)

                #use random network distillation
                if self.use_rnd:
                    rnd_reward = self.get_rnd_reward(obs[0])
                    reward += self.rnd_scale * rnd_reward
    

                l_obs_x = torch.cat([l_obs_x, prev_obs[0]], dim=0)
                l_obs_one_hot = torch.cat([l_obs_one_hot, prev_obs[1]], dim=0)
                l_rew = torch.cat([l_rew, torch.Tensor(np.array(1.*reward)).unsqueeze(0)], dim=0)
                l_act = torch.cat([l_act, act], dim=0)
                l_next_obs_x = torch.cat([l_next_obs_x, obs[0]], dim=0)
                l_next_obs_one_hot = torch.cat([l_next_obs_one_hot, obs[1]], dim=0)
                l_done = torch.cat([l_done,torch.Tensor(np.array(1.0*done)).unsqueeze(0)], dim=0)


                if self.get_depth_frame:
                    temp = self.pre_process(torch.Tensor(observation["depth"].copy()).unsqueeze(0), 25.0)

                    l_depth_frame = torch.cat([l_depth_frame, temp], dim=0)

                if self.get_class_frame:

                    temp = self.pre_process(torch.Tensor(observation["class_frame"]\
                            .copy()).unsqueeze(0).permute(0,3,1,2), 255.0)

                    l_class_frame = torch.cat([l_class_frame, temp], dim=0)

                if self.get_object_frame:
                    # not currently reaching this code (probably won't use this object_frame for the contest)
                    temp = self.pre_process(torch.Tensor(observation["object_frame"]\
                            .copy()).unsqueeze(0), 255.0)
                    l_object_frame = torch.cat([l_object_frame, temp], dim=0)

                l_class_labels = torch.cat([l_class_labels,\
                        torch.Tensor(observation["class_labels"]).unsqueeze(0)], dim=0)


            print("number of episodes {} with max/mean reward {:.2f}/{:.2f}"\
                    .format(torch.sum(l_done), torch.max(l_rew), torch.sum(l_rew)/torch.sum(l_done)))
            return l_obs_x, l_obs_one_hot, l_rew, l_act,\
                    l_next_obs_x, l_next_obs_one_hot, l_done, \
                    l_class_labels, l_depth_frame, l_class_frame,\
                    l_object_frame, l_info

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



    def train(self, max_epochs=1024, rollouts=True, train_policy=True):


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
                num_samples = min(len_dir_list, 20)

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
                                l_done[batch:batch+self.batch_size])

                        loss.backward()
                        smooth_loss = 0.9 * smooth_loss + 0.1 * loss.detach().numpy()

                        optimizer.step()

                self.losses.append(smooth_loss)
                self.epsilon = np.max([self.min_epsilon, self.epsilon*self.epsilon_decay])

                # update target network every once in a while
                if epoch % self.update_qt_every == 0:

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



        
def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """
    if n<=1:
        return "child"

    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
          MKL_NUM_THREADS="1",
          OMP_NUM_THREADS="1",
          IN_MPI="1"
        )
        print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpirun", "-np", str(n), sys.executable] \
            +['-u']+ sys.argv, env=env)
        return "parent"
    else:
        global nWorker, rank
        nWorker = comm.Get_size()
        rank = comm.Get_rank()
        #print('assigning the rank and nworkers', nWorker, rank)
        return "child"

def mantle(args):

    agent_fn = MyTempAgent
    agent = MyTempAgent()

    #pretrained = OffTaskModel()
    #pretrained.load_state_dict(torch.load("temp_off_task_model.pt"))
    #agent.feature_extractor.load_state_dict(pretrained.feature_extractor.state_dict())
    env = RobothorChallengeEnv(agent=agent)
    dqn = DQN(env, agent_fn, use_rnd=True)

    dqn.q.load_state_dict(torch.load("./my_dqn_temp.pt", map_location=torch.device("cpu")))
    dqn.qt.load_state_dict(torch.load("./my_dqn_temp.pt", map_location=torch.device("cpu")))

    # run rollouts and training in parallel, 
    # number of training epochs between qt updates = num_workers
    dqn.update_qt_every = 2

    t0 = time.time()
    epochs = 100
    num_workers = args.num_workers

    for epoch in range(epochs):
        bb = 0
        total_steps =0

        t1 = time.time()

        while bb <= num_workers: # - nWorker:
            pop_left = num_workers - bb
            for cc in range(1, min(nWorker, 1+pop_left)):
                comm.send([dqn.q, dqn.qt, dqn.epsilon], dest=cc)


            bb += cc

        print("mantle worker {} begin training".format(rank))
        dqn.train(max_epochs=1, rollouts=False, train_policy=True)
        print("mantle asking workers for go-continue")

        bb = 0
        while bb <= num_workers: # - nWorker:
            pop_left = num_workers - bb

            for cc in range(1, min(nWorker, 1+pop_left)):
                do_continue = comm.recv(source=cc)

            bb += cc

        print("continuation confirmation rec'd, saving policy")
        torch.save(dqn.q.state_dict(),"./my_dqn_temp.pt")


    # send signal to shutdown workers
    for cc in range(1,nWorker):
        comm.send([0, 0], dest=cc)
    data = 0

def arm(args):

    agent_fn = MyTempAgent
    agent = MyTempAgent()

    pretrained = OffTaskModel()
    pretrained.load_state_dict(torch.load("temp_off_task_model.pt"))
    agent.feature_extractor.load_state_dict(pretrained.feature_extractor.state_dict())
    env = RobothorChallengeEnv(agent=agent)
    dqn = DQN(env, agent_fn, use_rnd=True)


    while True:


        print("worker {} waiting for policy...".format(rank))
        [dqn.q, dqn.qt, dqn.epsilon] = comm.recv(source=0)

        if dqn.q == 0:
            print("worker {} shutting down".format(rank))
            break

        print("worker {} begin rollout".format(rank))
        dqn.train(max_epochs = 1, rollouts=True, train_policy=False)

        comm.send(True, dest=0)

    #comm.send(False, dest=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
                "test mpi for running multiple policies")
    parser.add_argument('-n', '--num_workers', type=int, \
            help="number off cores to use, default 8", default=7)

    args = parser.parse_args()
    num_workers = args.num_workers

    if mpi_fork(args.num_workers+1) == "parent":
        os._exit(0)

    if rank == 0:
        mantle(args)
    else:
        arm(args)

        


