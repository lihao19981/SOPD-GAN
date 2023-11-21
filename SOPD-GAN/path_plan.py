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


##轨迹规划
robot_traj =np.zeros((100,3))
rob_chose_vel = np.zeros((100,3))

k1 = 1
k2 = 0.2
k3 = 1
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




def path_plan(robot_po):

    ref_pos_ = np.zeros((1,3))
    ref_pos_[0][0] = robot_po[0][0]

    robot_pos_ = np.zeros((1,3))
    robot_pos_[0][0] = robot_po[0][0]
    robot_pos_[0][1] = robot_po[0][1]
    robot_pos_[0][2] = robot_po[0][2]

    robot_traj_ = np.zeros((100, 3))
    robot_traj_[0] = robot_pos_

    rob_vel = np.zeros((100,1))
    poss = 1

    while (robot_pos_[0][0] < goal[0] | poss <100):
        chose_ = movecontrol(vel_sample, robot_pos_, target_pos, safe_dis, ref_pos_, ref_vel, goal, con_value, weight,
                             score)
        robot_pred_ = np.zeros((pred_len, 3))
        robot_pred_[0] = robot_pos_

        local_pos = -1
        for j in range(1, pred_len):
            robot_pred_[j][2] = vel_sample[chose_][1] + robot_pred_[j - 1][2]
            robot_pred_[j][0] = math.cos(robot_pred_[j][2]) * vel_sample[chose_][0] + robot_pred_[j - 1][0]
            robot_pred_[j][1] = math.sin(robot_pred_[j][2]) * vel_sample[chose_][0] + robot_pred_[j - 1][1]
            if ((math.sqrt((robot_pred_[j][0] - goal[0]) ** 2 + (robot_pred_[j][1] - goal[1]) ** 2) < 0.2) | (
                    robot_pred_[j][0] > goal[0])):
                local_pos = j
                break
        if (local_pos == -1):
            robot_pos_[0] = robot_pred_[1]
            robot_traj_[poss] = robot_pos_
            rob_vel[poss] =chose_
            ref_pos_[0][0] = robot_pred_[1][0]
            poss = poss + 1
        else:
            for k in range(local_pos + 1):
                robot_traj_[poss] = robot_pred_[k]
                rob_vel[poss] = chose_
                poss += 1
            break
    return robot_traj_ , rob_vel

def control_mode():
    index = 0
    delta_x = 0.2

    ##定位参考点
    for i in range(100):
        if ((robot_pos[0][0] + delta_x) < robot_traj[i][0]):
            index = i
            break

    ##轨迹控制参数
    x_d = robot_traj[index][0]
    y_d = robot_traj[index][1]
    slope = (robot_traj[index + 1][1] - robot_traj[index][1]) / (robot_traj[index + 1][0] - robot_traj[index][0])
    theta_d = math.atan(slope)
    v_d = vel_sample[int(rob_chose_vel[index])][0] - 0.1
    w_d = vel_sample[int(rob_chose_vel[index])][1]

    ##轨迹控制
    x_e = (x_d - robot_pos[0][0]) * math.cos(robot_pos[0][2]) + (y_d - robot_pos[0][1]) * math.sin(robot_pos[0][2])
    y_e = (x_d - robot_pos[0][0]) * math.sin(robot_pos[0][2]) + (y_d - robot_pos[0][1]) * math.cos(robot_pos[0][2])
    theta_e = theta_d - robot_pos[0][2]

    cmd_v = v_d * math.cos(theta_e) + k2 * x_e
    cmd_w = w_d + k1 * v_d * y_e + k3 * math.sin(theta_e)


    return cmd_v ,cmd_w


def pred_callpack(data):
    global target_pos
    global robot_traj
    global rob_chose_vel
    for i in range(12):
        target_pos[i][0] = data.poses[i].pose.position.x
        target_pos[i][1] = data.poses[i].pose.position.y

    robot_traj = np.zeros((100,3))
    [robot_traj,rob_chose_vel ] = path_plan()

def lidar_callback(data):
    #   print(data.poses[-1].pose.position.x)
    global robot_pos
    robot_pos[0][0] = data.pose.position.x
    robot_pos[0][1] = data.pose.position.y

    ####四元数
    x = data.pose.orientation.x
    y = data.pose.orientation.y
    z = data.pose.orientation.z
    w = data.pose.orientation.w

    angler = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    anglep = math.asin(2 * (w * y - z * z))
    angley = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    robot_pos[0][2] = angley

def odom_callback(data):
    #   print(data.poses[-1].pose.position.x)
    global robot_pos
    robot_pos[0][0] = data.pose.pose.position.x
    robot_pos[0][1] = data.pose.pose.position.y

    ####四元数
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w

    angler = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    anglep = math.asin(2 * (w * y - z * z))
    angley = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    robot_pos[0][2] = angley


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same node are launched, the previous one is kicked off.
    # The anonymous=True flag means that rospy will choose a unique name for our 'listener' node so that multiple listeners can run simultaneously.
    rospy.init_node('control_node', anonymous=True)


    cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    # rospy.Subscriber('trajectory', Path , callback)
    # rospy.Subscriber('robot_pose_ekf/odom_combined', PoseWithCovarianceStamped, odom_callback)
    rospy.Subscriber('/slam_out_pose', PoseStamped, lidar_callback)
    rospy.Subscriber('pred_traj', Path, pred_callpack)
    #load model
    #pub = rospy.Publisher('chatter', String, queue_size=10)
    # spin() simply keeps python from exiting until this node is stopped

    rate = rospy.Rate(10)  # 10hz
    vel = Twist()

    while not rospy.is_shutdown():


        [vel.linear.x , vel.angular.z ] = control_mode()
        cmd_pub.publish(vel)
        #pub.publish(hello_str)  # 发布信息到主题
        rate.sleep()

        #####终止条件
        if (robot_pos[0][0]> goal[0]):
            vel.linear.x = 0
            vel.angular.z = 0
            cmd_pub.publish(vel)
            break

    rospy.spin()


if __name__ == '__main__':

    [robot_traj, rob_chose_vel] = path_plan()

    listener()

