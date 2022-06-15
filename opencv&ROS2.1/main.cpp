//
// Created by dzl on 22-6-14.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//std::string file_path = "/home/dzl/CLionProjects/-SLAM/opencv&ROS2.1/pic/Gnome_G018_HD_NoLogo.png";
int main(int argc, char **argv){
    cv::Mat image;
    image = cv::imread(argv[1],-1);
    if(image.data == nullptr){
        std::cerr<<"文件"<<argv[1]<<"不存在"<<std::endl;
        return 0;
    }
    std::cout<<"图像宽为："<<image.cols<<"，高为："<<image.rows<<"，通道数为："<<image.channels()<<std::endl;
    cv::imshow("image",image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}