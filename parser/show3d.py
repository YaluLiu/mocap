#!/usr/bin/env python
# coding: utf-8

# In[1]:


# packages for 3d interactivate visualization
import plotly
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
from   mpl_toolkits.mplot3d import proj3d
import plotly.graph_objs as go 
import numpy as np


# In[ ]:


def get_line_3d_segs(Nodes):
    
    line_3d_segs = []

    for node in Nodes.values():
        if node.parent == None:
            child = node.pos
            #parent = node.parent.pos
            root = go.Scatter3d(x = [child[0]], 
                         y = [child[1]],
                         z = [child[2]], 
                         marker = dict(size=4, color="red"),
                         name = node.name)
            line_3d_segs.append(root)
            continue

        child = node.pos
        parent = node.parent.pos
        trace = go.Scatter3d(
                        x = [child[0], parent[0]], 
                        y = [child[1], parent[1]], 
                        z = [child[2], parent[2]],
                        marker = dict(size=2, color="blue"),
                        line = dict(color="black", width=5),
                        name=node.name)
        line_3d_segs.append(trace)
        
    return line_3d_segs


# In[2]:


def generate_frames_layout(Nodes_lst):
    frames = []

    for k in range(len(Nodes_lst)):
        frames.append(dict(data = get_line_3d_segs(Nodes_lst[k]), name="frame{}".format(k+1)))    
    
    sliders=[
        dict(
            steps=[dict(method='animate',
                        args= [['frame{}'.format(k + 1)],
                                dict(mode='immediate',
                                     frame= dict(duration=10, redraw= True),
                                     transition=dict(duration=0))],
                                label='{:d}'.format(k+1)) for k in range(len(Nodes_lst))], 
                        transition= dict(duration=0),
                    x=0,
                    y=0, 
                    currentvalue=dict(font=dict(size=12), 
                                      prefix='slice: ', 
                                      visible=True, 
                                      xanchor='center'),  
                    len=1.0)]
    
    x_range = [-130, 130]
    y_range = [-130, 130]
    z_range = [-130, 130]
    t_sum = x_range[1]-x_range[0] + y_range[1]-y_range[0] + z_range[1]-z_range[0]
    ratio = ((x_range[1]-x_range[0])/t_sum, (y_range[1]-y_range[0])/t_sum, (z_range[1]-z_range[0])/ t_sum)
    ratio = np.array(ratio)*3
    layout = dict(
                width=800,
                height=700,
                autosize=False,
                title='action animation',
                scene=dict(
                    camera=dict(
                        up=dict(
                            x=0,
                            y=1,
                            z=0
                        ),
                        eye=dict(
                            x=-1.7428,
                            y=1.0707,
                            z=0.7100,
                        )
                    ),
                    xaxis=dict(
                        range=x_range),
                    yaxis=dict(
                        range=y_range),
                    zaxis=dict(
                        range=z_range),
                    aspectratio = dict( x=ratio[0], y=ratio[1], z=ratio[2] ),
                ),
                updatemenus=[
                    dict(type='buttons',
                         showactive=False,
                         y=1,
                         x=1,
                         xanchor='right',
                         yanchor='top',
                         pad=dict(t=0, r=10),
                         buttons=[dict(label='Play',
                                       method='animate',
                                       args=[None])])
                ],
                sliders=sliders
            )
    return frames, layout

def get_line_2d_segs(Nodes):
    line_2d_segs = []

    for node in Nodes.values():
        if node.parent == None:
            child = node.pos
            #parent = node.parent.pos
            root = go.Scatter(x = [child[0]], 
                         y = [child[1]],
                         marker = dict(size=4, color="red"),
                         name = node.name)
            line_2d_segs.append(root)
            continue

        child = node.pos
        parent = node.parent.pos
        trace = go.Scatter(
                        x = [child[0], parent[0]], 
                        y = [child[1], parent[1]], 
                        marker = dict(size=2, color="blue"),
                        line = dict(color="black", width=5),
                        name=node.name)
        line_2d_segs.append(trace)
        
    return line_2d_segs

def generate_frames_layout_2d(Nodes_lst, x_range = [550,650], y_range = [450,550]):
    frames = []

    for k in range(len(Nodes_lst)):
        frames.append(dict(data = get_line_2d_segs(Nodes_lst[k]), name="frame{}".format(k+1)))      
    
    sliders=[
        dict(
            steps=[dict(method='animate',
                        args= [['frame{}'.format(k + 1)],
                                dict(mode='immediate',
                                     frame= dict(duration=70, redraw= True),
                                     transition=dict(duration=0))],
                                label='{:d}'.format(k+1)) for k in range(len(Nodes_lst))], 
                        transition= dict(duration=0),
                    x=0,
                    y=0, 
                    currentvalue=dict(font=dict(size=12), 
                                      prefix='slice: ', 
                                      visible=True, 
                                      xanchor='center'),  
                    len=1.0)]
    
    layout = dict(
                width=800,
                height=700,
                autosize=False,
                title='action animation',              
                xaxis=dict(
                    autorange=False,
                    range=x_range),
                yaxis=dict(
                    autorange=False,
                    range=y_range),
                updatemenus=[
                    dict(type='buttons',
                         showactive=False,
                         y=1,
                         x=1.3,
                         xanchor='right',
                         yanchor='top',
                         pad=dict(t=0, r=10),
                         buttons=[dict(label='Play',
                                       method='animate',
                                       args=[None])])
                ],
                sliders=sliders
            )
    return frames, layout


# In[ ]:


def draw(Nodes,max_coordinate):
    fig = plt.figure(figsize=(15,15))
    ax_2 = fig.add_subplot(111, projection = '3d')
    ax = Axes3D(fig)

    ax.set_aspect("equal")
    ax.set_xlim3d(-max_coordinate, max_coordinate)
    ax.set_ylim3d(-max_coordinate, max_coordinate)
    ax.set_zlim3d(-max_coordinate, max_coordinate)

    xs, ys, zs = [], [], []
    for num in Nodes:
      node = Nodes[num]
      xs.append(node.pos[0])
      ys.append(node.pos[1])
      zs.append(node.pos[2])
    #x2, y2, _ = proj3d.proj_transform(node.pos[0],node.pos[1],node.pos[2], ax.get_proj())
    #ax_2.annotate(node.index, xy=(x2, y2))
    plt.plot(zs, xs, ys, 'b.')

    for num in Nodes:
      child = Nodes[num]
      parent = child.parent
      if parent is not None:
        xs = [child.pos[0], parent.pos[0]]
        ys = [child.pos[1], parent.pos[1]]
        zs = [child.pos[2], parent.pos[2]]
        plt.plot(zs, xs, ys, 'r')
    plt.show()


# In[ ]:




