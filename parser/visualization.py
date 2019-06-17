# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
from common.camera import *
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R

# packages for 3d interactivate visualization
import plotly
import plotly.graph_objs as go 


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)

def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    
    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']
    
    i = 0
    with sp.Popen(command, stdout = sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w*h*3)
            if not data:
                break
            i += 1
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
            if i == limit:
                break             
    
def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

def display_2D_skeleton(keypoints, skeleton, img=None):
    line_segs = []
    cols = []
    for j, j_parent in enumerate(skeleton.parents()):    
        if j_parent == -1:
            continue
    
        if len(skeleton.parents()) == keypoints.shape[0]:
            col = 'red' if j in skeleton.joints_right() else 'black'
            cols.append(col)  
            line_segs.append([(keypoints[j, 0], keypoints[j, 1]), (keypoints[j_parent, 0], keypoints[j_parent, 1])])
            
    lc = LineCollection(line_segs, colors = cols, linewidths=2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')
    ax.scatter(keypoints[10, 0], keypoints[10, 1],color="g")
    if img is None:
        # xlims, ylims
        top = keypoints[:,1].min()
        left = keypoints[:,0].min()
        bottom = keypoints[:,1].max()
        right = keypoints[:,0].max()
        ax.set_ylim(top, bottom)
        ax.set_xlim(left, right)
        ax.invert_yaxis()
        
    else:
        ax.imshow(img)

    ax.add_collection(lc)
    
    plt.show()

def display_3D_skeleton(pos, skeleton, azim, camera_pos = None):
    line_3d_segs = []
    cols = []
    for j, j_parent in enumerate(skeleton.parents()):
        if j_parent == -1:
            continue

        col = 'red' if j in skeleton.joints_right() else 'black'
        cols.append(col)
        line_3d_segs.append([(pos[j, 0], pos[j, 1], pos[j, 2]), (pos[j_parent, 0], pos[j_parent, 1], pos[j_parent, 2])])
        
    lc = Line3DCollection(line_3d_segs, colors=cols, linewidths=2)
    fig = plt.figure(figsize=(10,10))
    ax_3d = fig.add_subplot(1,1,1,projection='3d')
    ax_3d.add_collection3d(lc, zdir='z')
    ax_3d.view_init(elev=15., azim=azim)
    # ax_3d.set_aspect('equal')
    ax_3d.set_xlim3d([-4, 5])
    ax_3d.set_ylim3d([-4, 5])
    ax_3d.set_zlim3d([-4, 5])

    if camera_pos is not None:
        ax_3d.scatter(camera_pos["x"], camera_pos["y"], camera_pos["z"], color="g", s=50)
    
    plt.show()

def render_animation(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index+2, projection='3d')
        ax.view_init(elev=15., azim=azim) #  ‘elev’ stores the elevation angle in the z plane. ‘azim’ stores the azimuth angle in the x,y plane.
        ax.set_xlim3d([-radius/2, radius/2]) 
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
    
    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # Update 2D poses
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                    
                if len(parents) == keypoints.shape[1]:
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

            points = ax_in.scatter(*keypoints[i].T, 5, color='red', edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if len(parents) == keypoints.shape[1]:
                    lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                           [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j-1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j-1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j-1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')

            points.set_offsets(keypoints[i])
        
        print('{}/{}      '.format(i, limit), end='\r')
        

    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()

def display_2D_skeleton_beta(ax, keypoints, skeleton, img=None):
    line_segs = []
    cols = []
    for j, j_parent in enumerate(skeleton.parents()):    
        if j_parent == -1:
            continue
    
        if len(skeleton.parents()) == keypoints.shape[0]:
            col = 'red' if j in skeleton.joints_right() else 'black'
            cols.append(col)  
            line_segs.append([(keypoints[j, 0], keypoints[j, 1]), (keypoints[j_parent, 0], keypoints[j_parent, 1])])
            
    lc = LineCollection(line_segs, colors = cols, linewidths=2)
    
    ax.scatter(keypoints[:,0], keypoints[:,1], c="y")
    
    if img is None:
        # xlims, ylims
        top = keypoints[:,1].min() - 0.1
        left = keypoints[:,0].min() - 0.1
        bottom = keypoints[:,1].max() + 0.1
        right = keypoints[:,0].max() + 0.1
        ax.set_ylim(top, bottom)
        ax.set_xlim(left, right)
        ax.invert_yaxis()
        
    else:
        ax.imshow(img)

    ax.add_collection(lc)
    
    return ax

def display_3D_skeleton_beta(ax_3d, pos, skeleton, cams=None):
    line_3d_segs = []
    cols = []
    for j, j_parent in enumerate(skeleton.parents()):
        if j_parent == -1:
            continue

        col = 'red' if j in skeleton.joints_right() else 'black'
        cols.append(col)
        line_3d_segs.append([(pos[j, 0], pos[j, 1], pos[j, 2]), (pos[j_parent, 0], pos[j_parent, 1], pos[j_parent, 2])])
        
    lc = Line3DCollection(line_3d_segs, colors=cols, linewidths=2)
    
    ax_3d.add_collection3d(lc, zdir='z')      
    ax_3d.view_init(elev=15., azim=60)
#    ax_3d.set_aspect('equal')
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_xlim3d([-4, 6])
    ax_3d.set_ylim3d([-4, 6])
    ax_3d.set_zlim3d([-4, 6])
#    ax_3d.auto_scale_xyz([-4, 6],[-4, 6],[-4, 6])

    ax_3d.dist = 9
    
    if cams is not None:
        for cam in cams:
            cam_pos = camera_to_world(np.zeros((1,3), dtype="float32"), R=cam['orientation'], t=cam['translation'])
            # r = R.from_quat(cam["orientation"])
            # rmat = r.as_dcm()
            # cam_pos2 = np.matmul(-np.linalg.inv(rmat), cam["translation"].reshape(3,1))

            ax_3d.scatter(cam_pos[0][0], cam_pos[0][1], cam_pos[0][2], color="g", s=10)
            #ax_3d.scatter(cam_pos2[0], cam_pos2[1], cam_pos2[2], color="b", s=50)
            
    
    ax_3d.scatter(0, 0, 0, color="y", s=10)
    
    return ax_3d

def get_line_3d_segs(pts, k, skeleton, human36m_kpts_name):
    line_3d_segs = []
    cols = []
    for j, j_parent in enumerate(skeleton.parents()):
        if j_parent == -1:
            continue

        col = 'red' if j in skeleton.joints_right() else 'black'
        cols.append(col)
        trace = go.Scatter3d(
                    x = [pts[k, j, 0], pts[k, j_parent, 0]], 
                    y = [pts[k, j, 1], pts[k, j_parent, 1]], 
                    z = [pts[k, j, 2], pts[k, j_parent, 2]],
                    marker = dict(size=2, color="blue"),
                    text = [human36m_kpts_name[j]],
                    line = dict(color=col, width=5))
        line_3d_segs.append(trace)
    
    #if k != 0:
    trajectory = go.Scatter3d(
                    x = pts[:k, 0, 0], 
                    y = pts[:k, 0, 1], 
                    z = pts[:k, 0, 2],
                    marker = dict(color='#1f77b4', size=1),
                    line = dict(color='#1f77b4', width=1))
    line_3d_segs.append(trajectory)
        
    return line_3d_segs

def generate_frames_layout(pts, skeleton, human36m_kpts_name):
    frames = []

    for k in range(len(pts)):
        frames.append(dict(data = get_line_3d_segs(pts, k, skeleton, human36m_kpts_name), name="frame{}".format(k+1)))    
    
    sliders=[
        dict(
            steps=[dict(method='animate',
                        args= [['frame{}'.format(k + 1)],
                                dict(mode='immediate',
                                     frame= dict(duration=70, redraw= True),
                                     transition=dict(duration=0))],
                                label='{:d}'.format(k+1)) for k in range(len(pts))], 
                        transition= dict(duration=0),
                    x=0,
                    y=0, 
                    currentvalue=dict(font=dict(size=12), 
                                      prefix='slice: ', 
                                      visible=True, 
                                      xanchor='center'),  
                    len=1.0)]
    
    x_range = [-2, 5]
    y_range = [-2, 2]
    z_range = [-2, 5]
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

def reprojection(gt_cam, cam):
    return project_to_2d(torch.from_numpy(gt_cam.reshape(1, 17, 3)), torch.from_numpy(cam["intrinsic"].reshape(-1,9))).numpy()[0]

def display_trajectory(ax_traj, pos_3d, m):
    trajectory = np.array([[pos_3d[i,0,0], pos_3d[i,0,1], pos_3d[i,0,2], pos_3d[i+1,0,0], pos_3d[i+1,0,1], pos_3d[i+1,0,2],]for i in range(m)])
    ax_traj.scatter(trajectory[:,0], 
                    trajectory[:,1], 
                    trajectory[:,2], c = [i/len(trajectory[:,0]) for i in range(len(trajectory[:,0]))], cmap='cool', s = 5)
    
    return ax_traj