
#import robothor_challenge

from robothor_challenge import RobothorChallenge
from robothor_agents.agent import SimpleRandomAgent
from robothor_agents.agent import MyTempAgent

import ai2thor.controller
import ai2thor.util.metrics

import numpy as np

import torch
import torch.nn as nn


ALLOWED_ACTIONS =  ['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Stop']

class RobothorChallengeEnv(RobothorChallenge):


    def __init__(self, agent):
        super(RobothorChallengeEnv,self).__init__(agent)

        self.uninitialized = True

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


    def reset(self, get_depth_frame=False, get_class_frame=False, get_object_frame=False):
        # sample an episode randomly from self.episodes


        if (get_depth_frame or get_class_frame or get_object_frame) and self.uninitialized:


            self.setup_env()
            self.controller = ai2thor.controller.Controller(
                width=self.config['width'],
                height=self.config['height'],
                renderDepthImage=get_depth_frame,
                renderClassImage=get_class_frame,
                renderObjectImaeg=get_object_frame,
                **self.config['initialize']
            )

            self.uninitialized = False

        self.get_class_frame = get_class_frame
        self.get_depth_frame = get_depth_frame
        self.get_object_frame = get_object_frame

        num_epds = len(self.episodes)
        self.episode = self.episodes[np.random.randint(num_epds)]
        
        event = self.controller.last_event
        self.controller.initialization_parameters['robothorChallengeEpisodeId'] = self.episode['id']
        self.controller.reset(self.episode['scene'])
        teleport_action = dict(action='TeleportFull')
        teleport_action.update(self.episode['initial_position'])
        self.controller.step(action=teleport_action)
        self.controller.step(action=dict(action='Rotate', rotation=dict(y=self.episode['initial_orientation'], horizon=0.0)))
        self.total_steps = 0
        self.agent.reset()
        self.stopped = False


        obs = dict(object_goal=self.episode['object_type'], depth=event.depth_frame, rgb=event.frame)
        if self.get_class_frame:
            obs["class_frame"] = event.class_segmentation_frame
        if self.get_object_frame:
            obs["object_frame"] = event.object_segmentation_frame

        my_class_labels = [np.array([target.lower() in elem["name"].lower() and elem["visible"] \
                for elem in event.metadata["objects"]]).any() \
                for target in self.possible_targets]

        obs["class_labels"] = np.array(my_class_labels) * 1.0
        info = event.metadata

        info["target_nearby"] = False


        return obs, info


    def step(self, action ):

        self.total_steps += 1
        event = self.controller.last_event
        event.metadata.clear()
        
        if action not in ALLOWED_ACTIONS:
            raise ValueError('Invalid action: {action}'.format(action=action))

        event =  self.controller.step(action=action)

        obs = dict(object_goal=self.episode['object_type'], depth=event.depth_frame, rgb=event.frame)
        if self.get_class_frame:
            obs["class_frame"] = event.class_segmentation_frame
        if self.get_object_frame:
            obs["object_frame"] = event.object_segmentation_frame

        stopped = action == 'Stop'

        if stopped:
            simobj = self.controller.last_event.get_object(self.episode['object_id'])
            reward = 10.0 * simobj['visible']
            if reward == 10.0: print("winner!")
        else:
            reward = 0.0

        simobj = self.controller.last_event.get_object(self.episode['object_id'])
        if simobj['visible']: 
            reward += 0.1
            #print('Target object visible!!!')

        # reward for keeping on
        reward += 0.01

        done = stopped or self.total_steps >= self.config['max_steps']


        my_class_labels = [np.array([target.lower() in elem["name"].lower() and elem["visible"] \
                for elem in event.metadata["objects"]]).any() \
                for target in self.possible_targets]

        obs["class_labels"] = np.array(my_class_labels) * 1.0

        # give out extra info
        info = event.metadata
        info["target_nearby"] = self.target_nearby(info, obs)

        return obs, reward, done, info

    def target_nearby(self, info, obs):

        target_nearby = False

        for my_object in info["objects"]:

            # below is to avoid conflating e.g. "Baseball" with "BaseballBat"
            object_name = my_object["name"].split("_")[0].lower()

            if obs["object_goal"].lower() is object_name:
                import pdb; pdb.set_trace()
                #object matches target. Check if it is in view and nearby
                if my_object["visible"] and my_object["distance"] <= 1.0:
                    target_nearby = True

        return target_nearby

    def sample_action_space(self):
        pass



if __name__ == "__main__":

    # run a quick testV 

    print("instantiate agent and environment")
    #agent = SimpleRandomAgent()
    agent = MyTempAgent()
    env = RobothorChallengeEnv(agent=agent)
    
    done = False

    print("step through a few episodes in env")
    for trial in range(1000):
        obs = env.reset()
        done = False
        sum_rewards = 0.0
        import pdb; pdb.set_trace()
        while not done:

            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            sum_rewards += reward
        print("sum of rewards/done = {}/{}".format(sum_rewards, done), \
                obs['rgb'].shape, obs['object_goal'])
            
        
