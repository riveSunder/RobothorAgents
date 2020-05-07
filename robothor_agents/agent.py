from abc import ABC, abstractmethod
import random

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class Agent(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, observations):
        pass


class SimpleRandomAgent(Agent):

    def reset(self):
        pass

    def act(self, observations):
        # observations contains the following keys: rgb(numpy RGB frame), depth (None by default), object_goal(category of target object)
        action = random.choice(['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Stop'])
        return action

#class RandomDistillateNetwork(nn.Module):
#
#    def __init__(self):
#        super(RandomDistillateNetwork, self).__init__()
#        pass
#
#
#class AgentPolicy(nn.Module):
#
#    def __init__(self):
#        super(AgentPolicy, self).__init__() 
#        
#
#
#        self.block0 = nn.Sequential([nn.Conv2D])

#
#    def forward(self, x, one_hot=None, mode="pre"):
#        
#        # calculate action or depth, segmentation, and classification if pre-training
#
#        if mode == "pre":
#            pass
#            #return classes, segmentation, depth_map
#
#        elif mode == "act":
#            assert one_hot is not None, "Need to specify goal object"
#            pass
#            #return act


class MyTempAgent(Agent, nn.Module):

    def __init__(self, conv_depth=16):
        super(MyTempAgent, self).__init__()

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

        self.possible_actions =  ['MoveAhead', 'MoveBack', 'RotateRight', \
                'RotateLeft', 'LookUp', 'LookDown', 'Stop']

        self.conv_depth = conv_depth

        block0 = nn.Sequential(\
                nn.BatchNorm2d(3),\
                nn.Conv2d(3, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        #240x320
        block1 = nn.Sequential(\
                nn.Conv2d(conv_depth, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),\
                nn.MaxPool2d(kernel_size=2, stride=2))
    
        #120x160
        block2 = nn.Sequential(\
                nn.Conv2d(conv_depth, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),\
                nn.MaxPool2d(kernel_size=2, stride=2))
    
        #60x80
        block3 = nn.Sequential(\
                nn.Conv2d(conv_depth, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),\
                nn.MaxPool2d(kernel_size=2, stride=2))

        #30x40
        block4 = nn.Sequential(\
                nn.Conv2d(conv_depth, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),\
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_extractor = nn.Sequential(block0, block1, block2,\
                block3, block4)

        self.init_head()

    def init_head(self):

        conv_depth = self.conv_depth
        self.flat_head = True

        num_actions = 7
        num_objects = 12
        dense_in = 7*10*conv_depth+num_objects #15*20*conv_depth + num_objects
        dense_hid = 128

        dense_out = num_actions # number of possible actions

        self.flatten = nn.Flatten()

        self.dense_layers = nn.Sequential(\
                nn.Linear(dense_in, dense_hid),\
                nn.ReLU(),\
                nn.Linear(dense_hid, dense_hid),\
                nn.ReLU(),\
                nn.Linear(dense_hid, dense_out))

        

    def reset(self):
        pass

    def forward(self, x, one_hot=None, device=None):
        if device is not None:
            if x.device.type != device:
                x = x.to(torch.device(device))
                one_hot = one_hot.to(torch.device(device))


        x = self.feature_extractor(x)
        if self.flat_head:
            x = self.flatten(x)

        if one_hot is not None:
            x = torch.cat((x, one_hot), dim=1)
        x = self.dense_layers(x)

        # x is the value of each 
        return x

    def act(self, observations):
        img_input = torch.Tensor(observations['rgb'].copy()[np.newaxis,...]).reshape(1,3,480,640)

        target_str = observations['object_goal']

        target_one_hot = torch.Tensor(\
                np.array([1.0 if target_str in elem else 0.0 for elem in self.possible_targets]))\
                .reshape(1,12)


        # get action from forward policy
        softmax_logits = self.forward(img_input, target_one_hot)

        act = self.possible_actions[torch.argmax(softmax_logits)]

        return act

class ClassModel(MyTempAgent):

    def __init__(self, conv_depth=16):
        super(ClassModel, self).__init__()


    def init_head(self):

        conv_depth = self.conv_depth
        self.flat_head = True

        num_actions = 7
        num_objects = 12
        dense_in = 7*10*conv_depth
        dense_hid = 256

        dense_out = num_objects  # number of possible actions

        self.flatten = nn.Flatten()

        self.dense_layers = nn.Sequential(\
                nn.Linear(dense_in, dense_hid),\
                nn.ReLU(),\
                nn.Linear(dense_hid, dense_hid),\
                nn.ReLU(),\
                nn.Linear(dense_hid, dense_out),\
                nn.Sigmoid())
    

class DepthModel(MyTempAgent):

    def __init__(self, conv_depth=16):
        super(DepthModel, self).__init__()

    def init_head(self):

        conv_depth = self.conv_depth
        self.flat_head = False

        num_actions = 7
        num_objects = 12
        dense_in = 7*10*conv_depth+num_objects #15*20*conv_depth + num_objects
        dense_hid = 64


        block0 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=5, stride=2, padding=1),\
                nn.Tanh())

        #240x320
        block1 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(1,1)),\
                nn.Tanh())
    
        #120x160
        block2 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3,stride=2, padding=(0,1)),\
                nn.Tanh())
    
        #60x80
        block3 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(0,1)),\
                nn.Tanh())

        #30x40
        block4 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=4, stride=2, padding=(0,2)),\
                nn.Tanh())

        block5 = nn.Sequential(\
                nn.Conv2d(conv_depth, 1, kernel_size=3,stride=1, padding=1))

        self.dense_layers  = nn.Sequential(block0, block1, block2,\
                block3, block4, block5)



class SegmentationModel(MyTempAgent):

    def __init__(self, conv_depth=16):

        super(SegmentationModel, self).__init__()


    def init_head(self):

        conv_depth = self.conv_depth
        self.flat_head = False

        num_actions = 7
        num_objects = 12
        dense_in = 7*10*conv_depth+num_objects #15*20*conv_depth + num_objects
        dense_hid = 64


        block0 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=5, stride=2, padding=1),\
                nn.Tanh())

        #240x320
        block1 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(1,1)),\
                nn.Tanh())
    
        #120x160
        block2 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3,stride=2, padding=(0,1)),\
                nn.Tanh())
    
        #60x80
        block3 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(0,1)),\
                nn.Tanh())

        #30x40
        block4 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=4, stride=2, padding=(0,2)),\
                nn.Tanh())

        block5 = nn.Sequential(\
                nn.Conv2d(conv_depth, num_objects, kernel_size=3,stride=1, padding=1),\
                nn.LogSoftmax(dim=1))
        # use with nn.NLLLoss as objective function

        self.dense_layers  = nn.Sequential(block0, block1, block2,\
                block3, block4, block5)

class OffTaskModel(MyTempAgent):

    def __init__(self, conv_depth=16):
        super(OffTaskModel, self).__init__()

    def init_head(self):
        conv_depth = self.conv_depth
        self.flat_head = False

        num_actions = 7
        num_objects = 12
        dense_in = 7*10*conv_depth
        dense_hid = 64


        block0 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=5, stride=2, padding=1),\
                nn.Tanh())
        block1 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(1,1)),\
                nn.Tanh())
        block2 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3,stride=2, padding=(0,1)),\
                nn.Tanh())
        block3 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(0,1)),\
                nn.Tanh())
        block4 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=4, stride=2, padding=(0,2)),\
                nn.Tanh())
        block5 = nn.Sequential(\
                nn.Conv2d(conv_depth, 3, kernel_size=3,stride=1, padding=1))
        # use with nn.NLLLoss as objective function

        block10 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=5, stride=2, padding=1),\
                nn.Tanh())
        block11 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(1,1)),\
                nn.Tanh())
        block12 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3,stride=2, padding=(0,1)),\
                nn.Tanh())
        block13 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(0,1)),\
                nn.Tanh())
        block14 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=4, stride=2, padding=(0,2)),\
                nn.Tanh())
        block15 = nn.Sequential(\
                nn.Conv2d(conv_depth, 1, kernel_size=3,stride=1, padding=1))

        block20 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=5, stride=2, padding=1),\
                nn.Tanh())
        block21 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(1,1)),\
                nn.Tanh())
        block22 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3,stride=2, padding=(0,1)),\
                nn.Tanh())
        block23 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=3, stride=2, padding=(0,1)),\
                nn.Tanh())
        block24 = nn.Sequential(\
                nn.ConvTranspose2d(conv_depth, conv_depth, kernel_size=4, stride=2, padding=(0,2)),\
                nn.Tanh())
        block25 = nn.Sequential(\
                nn.Conv2d(conv_depth, 3, kernel_size=3,stride=1, padding=1))
        # use with nn.NLLLoss as objective function

        dense_out = num_objects  # number of possible actions
        self.flatten = nn.Flatten()

        self.layers_auto  = nn.Sequential(block0, block1, block2,\
                block3, block4, block5)
        self.layers_depth  = nn.Sequential(block10, block11, block12,\
                block13, block14, block15)
        self.layers_seg  = nn.Sequential(block20, block21, block22,\
                block23, block24, block25)
        
        self.layers_class = nn.Sequential(\
                nn.Linear(dense_in, dense_hid),\
                nn.ReLU(),\
                nn.Linear(dense_hid, dense_hid),\
                nn.ReLU(),\
                nn.Linear(dense_hid, dense_out),\
                nn.Sigmoid())
        # use sigmoid for multi-class classification


    def forward(self, x, device=None):

        if device is not None:
            if x.device.type != device:
                x = x.to(torch.device(device))
                one_hot = one_hot.to(torch.device(device))


        x = self.feature_extractor(x)

        x_flat = self.flatten(x)

        classes = self.layers_class(x_flat)
        depth = self.layers_depth(x)
        auto = self.layers_auto(x)
        seg = self.layers_seg(x)

        # x is the value of each 
        return depth, auto, seg, classes

if __name__ == "__main__":
    # run some tests

    my_agent = MyTempAgent()
    

    depth_model = DepthModel()
    seg_model = SegmentationModel()
    class_model = ClassModel()
    agent = MyTempAgent()
    off_task = OffTaskModel()

    temp = torch.randn(2,3,240,320)
    temp_one_hot = torch.randn(2,12)

    temp_res = depth_model(temp)

    print(temp_res.shape)
    temp_res = class_model(temp)

    print(temp_res.shape)
    temp_res = seg_model(temp)

    print(temp_res.shape)
    temp_res = agent(temp, temp_one_hot)

    print(temp_res.shape)

    temp_depth, temp_seg, temp_classes = off_task(temp)

    print(temp_depth.shape, temp_seg.shape, temp_classes.shape)
