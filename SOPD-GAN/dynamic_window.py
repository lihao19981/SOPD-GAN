import numpy as np
import math
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


target_circle = 0.5

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

for i in range(tot_num):
    vel_sample[i][0] = v_sam[i%(v_num+2)]
    vel_sample[i][1] = w_sam[i//(v_num+2)]

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
weight = [0,1,-2,0,0]



def movecontrol(vel_sample , robot_pos ,target_pos, safe_dis , ref_pos ,ref_vel , goal , con_value , weight , score ):
    pred_len = 8  ##预测步长
    chose = 0
    score_max = -1000.00
    for i in range(tot_num):
        if (vel_sample[i][1] != 0 ):
            if(abs(vel_sample[i][0] / vel_sample[i][1]) > 1.5):
                score[i] = -10000
                continue

        ###定义样本对应轨迹
        robot_pred = np.zeros((pred_len,3))
        robot_pred[0] = robot_pos
        for j in range(1, pred_len):
            robot_pred[j][2] = vel_sample[i][1] + robot_pred[j-1][2]
            robot_pred[j][0] = math.cos(robot_pred[j][2]) * vel_sample[i][0] + robot_pred[j-1][0]
            robot_pred[j][1] = math.sin(robot_pred[j][2]) * vel_sample[i][1] + robot_pred[j-1][1]

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
        for j in range(pred_len):
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



def pred_callpack(data):
    global target_pos
    for i in range(12):
        target_pos[i][0] = data.poses[i].pose.position.x
        target_pos[i][1] = data.poses[i].pose.position.y




def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same node are launched, the previous one is kicked off.
    # The anonymous=True flag means that rospy will choose a unique name for our 'listener' node so that multiple listeners can run simultaneously.
    rospy.init_node('control_node', anonymous=True)


    cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    # rospy.Subscriber('trajectory', Path , callback)
    rospy.Subscriber('robot_pose_ekf/odom_combined', PoseWithCovarianceStamped, odom_callback)
    rospy.Subscriber('pred_traj', Path, pred_callback)
    #load model
    #pub = rospy.Publisher('chatter', String, queue_size=10)
    # spin() simply keeps python from exiting until this node is stopped

    rate = rospy.Rate(10)  # 10hz
    vel = Twist()

    while not rospy.is_shutdown():

        chose_ = movecontrol(vel_sample, robot_pos, target_pos, safe_dis, ref_pos, ref_vel, goal, con_value, weight, score)
        vel.linear.x = vel_sample[chose_][0]
        vel.angular.z = vel_sample[chose_][1]
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

    a =  movecontrol(vel_sample, robot_pos, target_pos, safe_dis, ref_pos, ref_vel, goal, con_value, weight, score)

    listener()