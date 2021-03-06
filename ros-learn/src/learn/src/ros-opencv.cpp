//
// Created by dzl on 22-6-23.
//
// http://wiki.ros.org/image_transport

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <stdio.h>


int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_opencv");
    // 声明节点
    ros::NodeHandle nh;
    // image_transport image订阅和发布
    // image_transport ("raw") - The default transport, sending sensor_msgs/Image through ROS
    // 用上面声明的节点句柄初始化it，it和nh的功能基本一样使用it来发布和订阅相消息
    image_transport::ImageTransport it(nh);
    // 第一个参数是话题的名称，第二个是缓冲区的大小（消息队列的长度发布图像消息时消息队列的长度只能是1）
    image_transport::Publisher pub = it.advertise("image", 1);


    cv::VideoCapture cap("/home/dzl/CLionProjects/-SLAM/ros-learn/src/learn/200862413-1-64.flv");
    if (!cap.isOpened())
    {
        std::cerr << "Read video Failed !" << std::endl;
        return 0;
    }

    cv::Mat image;
    while(ros::ok() && cap.isOpened()){
        ROS_INFO("system is fine");
        cap.read(image);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        msg->header.stamp = ros::Time::now();
        ros::Rate loop_rate(5);
        pub.publish(msg);
        ros::spinOnce();
        // 通过睡眠度过一个循环中剩下的时间
        loop_rate.sleep();
    }
    cap.release();
}
//    = cv::imread("/home/dzl/CLionProjects/-SLAM/ros-learn/src/learn/Gnome_G018_HD_NoLogo.png",
//                               cv::IMREAD_COLOR);
//    if(image.empty()){
//        printf("image empty\n");
//    }
//    else
//        ROS_INFO("read image successfully");
// 将opencv格式的图像转化为ROS所支持的消息类型从而发布到相应的话题上，bgr8是编码格式


// 设置topic循环发布的频率为20hz


//    while (nh.ok()) {
//        pub.publish(msg);
//        ros::spinOnce();
//        // 通过睡眠度过一个循环中剩下的时间
//        loop_rate.sleep();
//    }