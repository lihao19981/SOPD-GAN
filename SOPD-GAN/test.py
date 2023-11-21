import numpy as np
import math
import time
import matplotlib
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Path
import argparse
import os
from attrdict import AttrDict


#####定义初始move参数
robot_pos = np.zeros((1,3))
v_max = 0.5
v_min = -0.1
w_max = 0.2
w_min = -0.2
robot_circle = 0.5

#####定义初始goal参数
goal = [10,0]

#####定义初始target参数
target_pos = np.array([[5.00 ,0.00] , [5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00] ,[5.00 ,0.00]] )

for i in range(12):
    target_pos[i] = target_pos[0]


target_circle = 0.4

safe_dis = robot_circle + target_circle

#####定义速度样本数
v_num = 11
w_num = 9
tot_num = (v_num+2) * (w_num+2)
vel_sample = np.zeros((tot_num,2))

v_sam = np.zeros((v_num + 2, 1))
w_sam = np.zeros((w_num + 2, 1))

for i in range(v_num+2):
    v_sam[i] = (v_max - v_min) * i / (v_num + 1) + v_min

for i in range(w_num+2):
    w_sam[i] = (w_max - w_min) * i / (w_num + 1) + w_min

local_pos = 0
for i in range(w_num+2):
    a=0
    for j in range((i * (v_num + 2)), ((i + 1) * (v_num + 2) )):
        vel_sample[j][0] = v_sam[a]
        a = a+1

    for j in range((i * (v_num+2) ) , ((i + 1) * (v_num+2) )):
        vel_sample[j][1] = w_sam[local_pos]
    local_pos += 1

######定义ref_robot参数
ref_pos = np.zeros((1,3))
ref_pos[0][2] = math.atan(goal[1] / goal[0])
ref_vel = 0.4


######定义动态窗口法参数
chose = 0
score = np.zeros((tot_num,1))
score_max = -100000

H = 0.00
S = 0.00
P = 0.00
T = 0.00
V = 0.00

con_value = [H,S,P,T,V]
weight = [0,1,-2,0,-1]

pred_len = 8


def movecontrol(vel_sample , robot_pos ,target_pos, safe_dis , ref_pos ,ref_vel , goal , con_value , weight , score ):
    pred_len = 8  ##预测步长
    chose = 0
    score_max = -1000.00
    for i in range(tot_num):
        if (vel_sample[i][1] != 0 ):
            if(abs(vel_sample[i][0] / vel_sample[i][1]) < 1.5):
                score[i] = -10000
                continue

        ###定义样本对应轨迹
        robot_pred = np.zeros((pred_len,3))
        robot_pred[0] = robot_pos
        for j in range(1, pred_len):
            robot_pred[j][2] = vel_sample[i][1] + robot_pred[j-1][2]
            robot_pred[j][0] = math.cos(robot_pred[j][2]) * vel_sample[i][0] + robot_pred[j-1][0]
            robot_pred[j][1] = math.sin(robot_pred[j][2]) * vel_sample[i][0] + robot_pred[j-1][1]

        ###定义ref_robot对应轨迹
        ref_traj = np.zeros((pred_len,3))
        ref_traj[0] = ref_pos
        for j in range(1, pred_len):
            ref_traj[j][0] = math.cos(ref_traj[j][2]) * ref_vel + ref_traj[j-1][0]
            ref_traj[j][1] = math.sin(ref_traj[j][2]) * ref_vel + ref_traj[j-1][1]

        #######指标计算

        ### H计算
        con_value[0] = abs(robot_pred[-1][2] - robot_pred[0][2])

        ### S计算
        S = 1000.000
        for j in range(1, pred_len):
            dis = math.sqrt((robot_pred[j][0] - target_pos[j][0])**2 + (robot_pred[j][1] - target_pos[j][1])**2)
            S = min(S , dis)
            if (S < safe_dis):
                break

        if (S < safe_dis):
            con_value[1] = -10000.00
        else:
            con_value[1]  = 0

        ### P计算
        P = 10000
        for j in range(1,pred_len):
            P = min(P,math.sqrt((ref_traj[j][0] - robot_pred[j][0])**2 + (ref_traj[j][1] - robot_pred[j][1])**2))
        con_value[2] = P

        ### T计算
        T = abs(ref_traj[pred_len -1][2] - robot_pred[pred_len -1][2])
        con_value[3] = T

        ### V计算
        V = math.sqrt((goal[0] - robot_pred[pred_len - 1][0])**2 + (goal[1] - robot_pred[pred_len - 1][1])**2)
        con_value[4] = V

        ####评分
        score[i] = 0.00
        for j in range(5):
            score[i] += weight[j] * con_value[j]
        if (score[i] > score_max ):
            score_max = score[i]
            chose = i

    return  chose





if __name__ == '__main__':

   start = time.clock()
   robot_traj =np.zeros((10000,3))
   poss = 1
   while (robot_pos[0][0] < goal[0]):
        chose_ = movecontrol(vel_sample, robot_pos, target_pos, safe_dis, ref_pos, ref_vel, goal, con_value, weight, score)
        robot_pred_ = np.zeros((pred_len, 3))
        robot_pred_[0] = robot_pos

        local_pos = -1
        for j in range(1, pred_len):
            robot_pred_[j][2] = vel_sample[chose_ ][1] + robot_pred_[j - 1][2]
            robot_pred_[j][0] = math.cos(robot_pred_[j][2]) * vel_sample[chose_ ][0] + robot_pred_[j - 1][0]
            robot_pred_[j][1] = math.sin(robot_pred_[j][2]) * vel_sample[chose_ ][0] + robot_pred_[j - 1][1]
            if((math.sqrt((robot_pred_[j][0] - goal[0])**2 + (robot_pred_[j][1] - goal[1])**2) < 0.2 )|( robot_pred_[j][0] >goal[0])):
                local_pos = j
                break
        if(local_pos == -1):
            robot_pos[0] =robot_pred_[1]
            robot_traj[poss] = robot_pos
            ref_pos[0][0] = robot_pred_[1][0]
            poss = poss + 1
        else:
            for k in range(local_pos+1):
                robot_traj[poss] = robot_pred_[k]
                poss += 1
            break

   end = time.clock()
   print(end - start)




   for i in range(poss):
        with open("test.txt", 'a', encoding="utf-8") as f:
            f.write(str(robot_traj[i][0]))
            f.write(' ')
            f.write(str(robot_traj[i][1]))
            f.write('\n')


