#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from bvh import Bvh, BvhNode
import math
import copy


# In[3]:


class MyNode():
    def __init__(self,name):
        #main attribute
        self.name = name
        self.parent = self.parent_index = self.pos = None
        #not main 
        #self.children = []
        self.offset = self.channels = self.index = None
        self.matrix = None


# In[4]:


def make_parent(joints):
    for num in range(0,len(joints)):
        joint = joints[num]
        if(joint.parent_index != None):
            parent_idx = joint.parent_index
            joint.parent = joints[parent_idx]
            #joints[parent_idx].children.append(joint)


# In[5]:


def get_order_joints(mocap):
    joints = mocap.get_joints()
    Nodes = {}
    for num in range(0,len(joints)):
        name = joints[num].value[1]
        node = MyNode(name)
        node.offset = np.array(mocap.joint_offset(name))
        node.channels = mocap.joint_channels(name)
        node.index = mocap.get_joint_index(name)
        if(num != 0):
            parent_name = joints[num].parent.value[1]
            node.parent_index = mocap.get_joint_index(parent_name)
        else:
            node.parent_index = None
        Nodes[num] = node
    make_parent(Nodes)
    return Nodes


# In[6]:


def get_motions(mocap,frame_idx):
    motions= []
    root_pos = mocap.frame_joint_channels(frame_idx, 'Hips', ['Xposition','Yposition','Zposition'])
    root_pos = np.array(root_pos)
    names = mocap.get_joints_names()
    for num in range(0,len(names)):
        name = names[num]
        motion = mocap.frame_joint_channels(frame_idx, name, ['Xrotation','Yrotation','Zrotation'])
        motions.append(motion)
    return motions,root_pos


# In[7]:


def euler_to_matrix(euler,axis = 'syxz'):
    rotation = np.deg2rad(euler)
    matrix = euler2mat(*rotation,axis)
    return matrix
def euler_to_matrix_myself(euler):
    rotation = np.deg2rad(euler)
    alpha = rotation[0]
    beta = rotation[1]
    gamma = rotation[2]
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    Rx = np.asarray([[1, 0, 0], 
          [0, ca, sa], 
          [0, -sa, ca]
          ])

    Ry = np.asarray([[cb, 0, -sb], 
          [0, 1, 0],
          [sb, 0, cb]])

    Rz = np.asarray([[cg, sg, 0],
          [-sg, cg, 0],
          [0, 0, 1]])

    rotmat = np.eye(3)

    rotmat = np.matmul(Ry, rotmat)
    rotmat = np.matmul(Rx, rotmat)
    rotmat = np.matmul(Rz, rotmat)
    return rotmat


# In[8]:


def compute_pos(Nodes,motions,root_pos):
    for num in range(0,len(Nodes)):
        node = Nodes[num]
        rot_matrix = euler_to_matrix_myself(motions[num])
        if(node.parent_index != None):
            node.matrix = rot_matrix.dot(node.parent.matrix)
            direction = node.offset.dot(node.parent.matrix)
            node.pos =  node.parent.pos + direction
        else:
            node.pos =  root_pos
            node.matrix = rot_matrix


# <font color=blue size=5> class of BVHParser</font>

# In[9]:


class BVHParser:
    def __init__(self, bvh_path):
        self.bvh_path = bvh_path
        self.joints_lst = []
        self.frames_num = self.joints_num = 0
    def parse(self):
        with open(self.bvh_path) as f:
            mocap = Bvh(f.read())
        self.frames_num = mocap.nframes
        joints = get_order_joints(mocap)
        self.joints_num = len(joints)
        self.joints_lst = []
        for frame_idx in range(self.frames_num):
            motions,root_pos = get_motions(mocap,frame_idx)
            compute_pos(joints,motions,root_pos)
            self.joints_lst.append(copy.deepcopy(joints))


