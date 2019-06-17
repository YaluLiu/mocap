asf_to_human = [
('root', 'Pelvis') ,
('rhipjoint', 'RHip') ,('rfemur', 'RKnee') ,('rtibia', 'RAnkle') ,
('lhipjoint', 'LHip') ,('lfemur', 'LKnee') ,('ltibia', 'LAnkle') ,
('upperback', 'Spine1') ,('lowerneck', 'Neck') ,('upperneck', 'Head') ,('head', 'Site') ,
('lclavicle', 'LShoulder') ,('lradius', 'LElbow') ,('lwrist', 'LWrist') ,
('rclavicle', 'RShoulder') ,('rradius', 'RElbow') ,('rwrist', 'RWrist') , 
]

bvh_to_human = [
('Hips', 'Pelvis') ,
('RightUpLeg', 'RHip') ,('RightLeg', 'RKnee') ,('RightFoot', 'RAnkle') ,
('LeftUpLeg',  'LHip') , ('LeftLeg', 'LKnee') , ('LeftFoot', 'LAnkle') ,
('Spine', 'Spine1') ,('Spine1', 'Neck') ,('Neck', 'Head') , ('Head', 'Site') ,
('LeftArm', 'LShoulder') ,('LeftForeArm', 'LElbow') ,('LeftHand', 'LWrist') ,
('RightArm', 'RShoulder') ,('RightForeArm', 'RElbow') ,('RightHand', 'RWrist') , 
]


c_to_parent = [
    (0,-1),
    (1,0),(2,1),(3,2),
    (4,0),(5,4),(6,5),
    (7,0),(8,7),(9,8),(10,9),
    (11,8),(12,11),(13,12),
    (14,8),(15,14),(16,15),
]



def get_joint_by_name(joint_name,joints):
    for joint in joints.values():
        if(joint_name == joint.name):
            return joint
    return None
def make_src_joints_names(trans_pairs):
    names = []
    for pair in trans_pairs:
        names.append(pair[0])
    return names
def trans_joints(names,src_joints):
    joints = dict()
    joints_num = len(names)
    #for get coordinate
    for idx,name in enumerate(names):
        joint = get_joint_by_name(name,src_joints)
        if(joint != None):
            joints[idx] = joint
        else:
            print("Error!")
    #make new parent tree
    for pair in c_to_parent:
        idx = pair[0]
        parent_idx = pair[1]
        joints[idx].index =  idx
        joints[idx].parent_index =  parent_idx
        if(parent_idx != -1):
            joints[idx].parent =  joints[parent_idx]
    return joints

def special_solve_bvh(joints_lst):
    for joints in joints_lst:
        tmp = joints[11].pos
        joints[9].pos = joints[10].pos
        joints[10].pos = joints[11].pos
        joints[11].pos = joints[12].pos - (joints[12].pos - tmp) / 2

def trans_joints_lst(src_joints_lst,old_type):
    if(old_type == 'bvh'):
        special_solve_bvh(src_joints_lst)
        names = make_src_joints_names(bvh_to_human)
    elif(old_type == 'amc' or old_type == 'asf' ):
        names = make_src_joints_names(asf_to_human)
    dst_joints_lst = []
    for src_joints in src_joints_lst:
        dst_joints = trans_joints(names,src_joints)
        dst_joints_lst.append(dst_joints)
    return dst_joints_lst
