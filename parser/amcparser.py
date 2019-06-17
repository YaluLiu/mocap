#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
from transforms3d.euler import euler2mat
from pprint import pprint
import math
import copy

# In[2]:


class Joint:
  cnt = 0
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
    self.name = name
    self.index = 0
    self.direction = np.reshape(direction, [3])
    self.length = length
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.pos = None
    self.matrix = None
    
  def set_index(self):
    if self.name == 'root':
        Joint.cnt = 0; self.index = Joint.cnt; Joint.cnt += 1
    else:
        self.index = Joint.cnt; Joint.cnt += 1
    for child in self.children:
        child.set_index()
  def compute_pos(self, motion):
    if self.name == 'root':
      self.pos = np.reshape(np.array(motion['root'][:3]), [3])
      rotation = motion['root'][3:]
      rotation = np.deg2rad(rotation)
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
    else:
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
      self.matrix = self.parent.matrix.dot(self.matrix)
      self.pos = self.parent.pos + self.length * self.matrix.dot(self.direction)
    for child in self.children:
      child.compute_pos(motion)

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


# In[3]:


def draw(joints):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      xs.append(joint.pos[0])
      ys.append(joint.pos[1])
      zs.append(joint.pos[2])
    plt.plot(zs, xs, ys, 'b.')

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.pos[0], parent.pos[0]]
        ys = [child.pos[1], parent.pos[1]]
        zs = [child.pos[2], parent.pos[2]]
        plt.plot(zs, xs, ys, 'r')
    plt.show()


# In[4]:


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


# In[5]:


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])
    #print(direction.shape)  (3,)
    
    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


# In[6]:


def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)
  return frames


# make no order joints to order-nodes

# In[7]:


def make_order(joints):
    nodes = {}
    for joint in joints.values():
        nodes[joint.index] = joint
    return nodes

# In[22]:


class AMCParser:
    def __init__(self, asf_path, amc_path):
        self.asf_path = asf_path
        self.amc_path = amc_path
        self.joints_lst = []
    def parse(self):
        #get nodes tree and set index
        no_order_joints = parse_asf(self.asf_path)
        no_order_joints['root'].set_index()
        joints = make_order(no_order_joints)
        #get root pos and eulers data
        motions = parse_amc(self.amc_path)
        self.joints_num = len(joints)
        self.frames_num = len(motions)
        #compute pos 
        self.joints_lst = []
        for frame_idx in range(self.frames_num):
            joints[0].compute_pos(motions[frame_idx])
            self.joints_lst.append(copy.deepcopy(joints))

    def test(self):
        print(self.num_nodes == 31) #size_node
        #self.nodes[0].compute_pos(self.eulers[0])
        #pos1 = np.array([ -2.36961433,14.08463284,-14.02630763])
        #pos2 = self.nodes[self.num_nodes - 1].pos
        #raise assert if not equal
        #np.testing.assert_array_almost_equal(pos1,pos2)


# main program

# In[23]:





# In[24]:


if __name__ == "__main__":
    asf_path = './data/16.asf.txt'
    amc_path = './data/16_01_jump.amc.txt'
    output_json_path = './Output/asf_points.json'
    output_npy_path = './Output/amc_points.npy'
    amcparser = AMCParser(asf_path,amc_path,output_json_path,output_npy_path)
    amcparser.parse()
    amcparser.save()
    amcparser.test()

# In[25]:








# In[ ]:





# In[ ]:




