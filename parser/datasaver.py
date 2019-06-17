import json
from pprint import pprint
import numpy as np


#save npy data of coordinate data
def get_points_data(nodes,dimension):
    num_points = len(nodes)
    pdata = np.zeros((num_points,dimension))
    for (idx,node) in nodes.items():
        pdata[idx] = node.pos            
    return pdata
def make_npdata(nodes_lst,file_name,dimension,special_solve = False):
    frames_num = len(nodes_lst)
    joints_num = len(nodes_lst[0])
    pdatas = np.zeros((frames_num,joints_num,dimension))
    for frame_idx in range(frames_num):
      pdatas[frame_idx] = get_points_data(nodes_lst[frame_idx],dimension)
    if(dimension == 3 and special_solve == True):
        root = pdatas[0][3].copy()
        pdatas -= root
        pdatas /= 15
    np.save(file_name,pdatas)
 
#save node tree to json file
def get_json_points(nodes):
    points = list()
    for (idx,node) in nodes.items():
        point = {}
        point["name"] = node.name
        point["index"] = node.index
        if(node.parent == None):
            point["parent"] = -1
        else:
            point["parent"] = node.parent.index
        points.append(point)
    return points
def make_json(nodes,output_npy_path):
    json_data = {}
    json_data["npy_path"] = output_npy_path
    points = get_json_points(nodes)
    json_data["skeleton_data"] = points
    return json_data
    

def save_data(node_lst,output_json_path, output_npy_path,dimension = 3,special_solve = False):
    make_npdata(node_lst,output_npy_path, dimension,special_solve)
    json_data = make_json(node_lst[0],output_npy_path)
    with open(output_json_path,"w") as file:
        json.dump(json_data,file)
