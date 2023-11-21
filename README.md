# SOPD-GAN

This is the code for the paper

[Pedestrian Trajectory Prediction Based on SOPD-GAN used for the Trajectory Planning and motion control of mobile robot](https://ieeexplore.ieee.org/document/10309117)

 Focusing on the application of mobile robots, the focus of this research is to enhance their performance in dynamic scenarios. To effectively plan the robot's path to avoid pedestrians, a machine learning algorithm is employed to predict the future trajectory of pedestrians, thus improving the accuracy of forecasting their multi-modal motion. The existing prediction methods primarily rely on pedestrian history and current movement attributes to predict future movement, they often overlook the impact of static obstacles on pedestrian movement decision. Therefore, in this study, a static obstacles probability description generative adversarial network (SOPD-GAN) is proposed. 
 
 The static obstacles probability description (SOPD) represents the future movement space of pedestrians and assessesthe likelihood of entry. Additionally, we incorporate pedestrian historical trajectory information using LSTM,and combine it with SOPD to form the generator model. The training of this model is carried out using a generative adversarial network (GAN), which is referred to as SOPD-GAN.
![image](https://github.com/lihao19981/SOPD-GAN/assets/53962474/1563dc8e-43a3-4053-be3b-66f188d31691)
![image](https://github.com/lihao19981/SOPD-GAN/assets/53962474/25625afe-3b64-42c5-b70b-4e585d99534c)
![image](https://github.com/lihao19981/SOPD-GAN/assets/53962474/7263578c-3116-4573-ba0f-88112e922ad6)
![image](https://github.com/lihao19981/SOPD-GAN/assets/53962474/7714ea80-f83a-4cc4-8fe1-9bf1eb42417c)
 ![image](https://github.com/lihao19981/SOPD-GAN/assets/53962474/5ecd7700-4f9f-4134-b2df-cf25d260c610)


# MODEL
![image](https://github.com/lihao19981/SOPD-GAN/assets/53962474/5850a6f7-1e8a-4ff7-b49d-c0d70c00ece7)

# Running Models

This model is developed for mobile robot platform and integrated with ROS.

ROS of the corresponding version is needed to install on the Ubuntu 18.04 operating system.
