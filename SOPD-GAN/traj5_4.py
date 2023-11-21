import numpy as np
import matplotlib
from sopd.data import trajectories
import torch
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import argparse
import os
from attrdict import AttrDict
from sopd.data.loader import data_loader
from sopd.models import TrajectoryGenerator
from sopd.losses import displacement_error, final_displacement_error
from sopd.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='./sopd-experiments/models/sopd-models',type=str)  #尝试直接更改参数路径 理论上该变量控制模型导入？
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

parser.add_argument('--showStatistics', default=0, type=int)
parser.add_argument('--use_gpu', default=0, type=int)

global tar_pos
tar_pos = 0

def get_generator(checkpoint, evalArgs):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,

        use_gpu=evalArgs.use_gpu)
    generator.load_state_dict(checkpoint['g_state'], strict = False)

    if evalArgs.use_gpu:
        generator.cuda()
    else:
        generator.cpu()

    generator.train()
    return generator

def modelpred(obs_traj, obs_traj_rel):
    # pred target traj

    pred_traj_rel, currPoolingStatistics = generator(obs_traj, obs_traj_rel, seq_start_end)
    pred_traj = relative_to_abs(
        pred_traj_rel, obs_traj[-1]
    )

    return pred_traj


def callback(data):

    #load target_tarj
    global tar_pos
    global pub
    rospy.loginfo("target pose: x:%0.6f, y:%0.6f ,pos:%0.6f", data.linear.x, data.linear.y, tar_pos)
    x = data.linear.x
    y = data.linear.y






    if tar_pos >7:
        obs_traj = target_[(tar_pos - 8):tar_pos]
        obs_traj_rel = target_rel[(tar_pos - 8):tar_pos]


        rospy.loginfo("obs_traj: x:%0.6f,y:%0.6f", float(obs_traj[0,0,0]), float(obs_traj[0,0,1]))
        pred_traj = modelpred(obs_traj, obs_traj_rel)


        pub_path = Path()

        for i in range(12):
            traj_point = PoseStamped()
            traj_point.header.frame_id = 'pred_traj'
            traj_point.header.seq = i
            traj_point.pose.position.x = float(pred_traj[i, 0, 0])
            traj_point.pose.position.y = float(pred_traj[i, 0, 1])
            pub_path.poses.append(traj_point)

        pub.publish(pub_path)

        print(target_)

    #load obs




if __name__ == '__main__':

    a = open('a.txt', mode='w')
    #load model
    evalArgs = parser.parse_args()
    if os.path.isdir(evalArgs.model_path):
        filenames = os.listdir(evalArgs.model_path)
        filenames.sort()
        paths = [
            os.path.join(evalArgs.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [evalArgs.model_path]

    path =  paths[0]
    sopd_model = torch.load(path, map_location='cpu')
    checkpoint = torch.load(path, map_location='cpu')
    generator = get_generator(checkpoint, evalArgs)
    _args = AttrDict(checkpoint['args'])

    path = './sopd-experiments/datasets/eth/test'

    _, loader = data_loader(_args, path)
    for batch in loader:

        if evalArgs.use_gpu:
            batch = [tensor.cuda() for tensor in batch]
        else:
            batch = [tensor.cpu() for tensor in batch]

        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         non_linear_ped, loss_mask, seq_start_end) = batch


    # load data
    target_ = torch.tensor([[[0.000,0.000]],[[1.000,0.000]],[[2.000,0.000]],[[3.000,0.000]],[[4.000,0.000]],[[5.000,0.000]],[[6.000,0.000]],[[7.000,0.000]]])
    target_[:,0]=obs_traj[:,1,:]
    target_rel = target_
    seq_start_end = torch.tensor([[0,0]])

    target_ = torch.zeros(20,1,2)
    target_rel = torch.zeros(20, 1, 2)
    target_rel[0,0,0] = 0.
    target_rel[0,0,1] = 0.


    traj_point = PoseStamped()
    traj_point.header.frame_id = 'pred_traj'
    traj_point.pose.position.x = float(obs_traj[0,0,0])

    obs_traj = torch.zeros(8,1,2)
    obs_traj_rel = torch.zeros(8, 1, 2)

    #get final pred
    pred_traj_fake_rel, currPoolingStatistics = generator(target_,target_rel,seq_start_end)
    pred_traj_fake = relative_to_abs(
        pred_traj_fake_rel, target_[-1]
    )
    pred_traj_fake_rel, currPoolingStatistics = generator(obs_traj,obs_traj_rel,seq_start_end)
    target = [0.4893,0.2968,0.6191,0.6294,1.0431,0.5862,1.2475,0.4628,1.6167 ,0.4481,1.9937 ,0.3429,2.0983 ,0.2124,2.4632 ,0.1813,2.9771 ,-0.0456,3.5448 ,-0.1948,3.8788 ,-0.2011,4.1716 ,-0.2722,4.3257 ,-0.2933,4.6363 ,-0.3101,4.9009 ,-0.3675,5.0351 ,-0.3614,5.8906 ,-0.4638,6.4832 ,-0.5209,7.1916 ,-0.5304,7.7096,-0.5221];
    # start ros_node
    target = [1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000]
    target = [0.991670683788255,0.0285597742393885,1.03118907056723,0.0288178669668428,1.05931316357993,0.0283830432332731,1.10142895447139,0.0289199818304786,1.12775276322368,0.0271368128342588,1.17397323884186,0.0281455134204602,1.20055768836479,0.0275050433581064,1.24762023966026,0.0289024059647750]
    for i in range(8):
        target_[tar_pos, 0, 0] = target[2*i]
        target_[tar_pos, 0, 1] = target[2*i+1]

        if tar_pos > 0:
            target_rel[tar_pos, 0, 0] = target_[tar_pos, 0, 0] - target_[tar_pos - 1, 0, 0]
            target_rel[tar_pos, 0, 1] = target_[tar_pos, 0, 1] - target_[tar_pos - 1, 0, 1]

        tar_pos = tar_pos + 1


    for i in range(1):

        print(modelpred(target_[i:i+8,:,:],target_rel[i:i+8,:,:]))
# 。。。。。。。。。。

