#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# packages for 3d interactivate visualization
import plotly
import plotly.graph_objs as go 
import copy



# ### 关键点class

# In[2]:

from parser.node import Node
from parser.show3d import generate_frames_layout,generate_frames_layout_2d

# In[3]:


class DataReader:
    def __init__(self,json_path):
        self.json_path = json_path
        self.npy = None
        self.frames_num = 0
        self.joints_num = 0
        self.frames = []
    def load_json(self):
        with open(self.json_path) as file:
            json_data = json.load(file)
        self.npy_path = json_data['npy_path']
        nodes = {}
        for point in json_data['skeleton_data']:
            node = Node(index = point['index'],name = point['name'],parent_index = point['parent'])
            nodes[node.index] = node
        for node in nodes.values():
            node.parent = None if node.parent_index == -1 else nodes[node.parent_index]
        return nodes
    def load_data(self):
        nodes = self.load_json()
        npy_data = np.load(self.npy_path)
        self.frames = []
        self.frames_num = npy_data.shape[0]
        self.joints_num = npy_data.shape[1]
        assert(self.joints_num == len(nodes))
        for frame_idx in range(0,self.frames_num):
            for node in nodes.values():
                node.pos = npy_data[frame_idx][node.index]
            self.frames.append(copy.deepcopy(nodes))
                
    def show_data(self,frames_num = -1,frames_distance = 1,dimension = 3, x_range = [-1,1], y_range = [-1,1]):
        frames_num = self.frames_num if frames_num == -1 else frames_num
        plotly.offline.init_notebook_mode(connected=True)
        if(dimension == 3):
            frames, layout = generate_frames_layout(self.frames[:frames_num:frames_distance])
        elif(dimension == 2):
            frames, layout = generate_frames_layout_2d(self.frames[:frames_num:frames_distance],x_range , y_range)
        fig = dict(data=frames[0]['data'], layout=layout, frames=frames)
        plotly.offline.iplot(fig)


# ## Load Json and npy data

# In[4]:
if __name__ == "__main__":
    json_path = './Output/asf_points.json'
    npy_path = './Output/amc_points.npy'
    # json_path = './Output/human_points.json'
    # npy_path = './Output/human_points.npy'
    reader = DataReader(json_path)
    reader.load_data()
    reader.show_data(1)
    print(reader.frames_num)
    print(reader.joints_num)




