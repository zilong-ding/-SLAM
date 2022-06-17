//
// Created by dzl on 22-6-14.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//std::string file_path = "/home/dzl/CLionProjects/-SLAM/opencv&ROS2.1/pic/Gnome_G018_HD_NoLogo.png";
int main(int argc, char **argv){
    cv::Mat image;
    image = cv::imread(argv[1],-1);
    cv::putText(image,"SLAM",cv::Point(50, 100), 0, 2, cv::Scalar(80, 25, 255));
    if(image.data == nullptr){
        std::cerr<<"文件"<<argv[1]<<"不存在"<<std::endl;
        return 0;
    }
    std::cout<<"图像宽为："<<image.cols<<"，高为："<<image.rows<<"，通道数为："<<image.channels()<<std::endl;
    std::cout<<"图像维度："<< image.dims << std::endl;
    std::cout<<"图像数据类型："<<image.type()<<std::endl;
    std::cout<<"图像数据："<<image.depth()<<std::endl;
    std::cout<<"image.elemsize::  "<<image.elemSize()<<std::endl;
    std::cout<<"image.step::  "<<image.step<<std::endl;
//    std::cout<<"image::  "<<image<<std::endl;


    for (int y = 0; y < image.rows; y++){
        uchar* row_ptr = image.ptr(y);// 定义一个uchar类型的row_ptr指向图像行的头指针
        for (int x = 0; x < image.cols*image.channels(); x++){
            // 遍历图像每一行所有通道的数据 Mat类矩阵矩阵中每一行中的每个元素都是挨着存放, 每一行中存储的数据数量为列数与通道数的乘积 即代码中指针向后移动cols*channels()-1位
            std::cout << "row_ptr[" << x << "]: " << (int)row_ptr[x] << std::endl;
        }
    }

    cv::imshow("image",image);
    cv::waitKey(0);
//    for (int i = 0; i < image.cols; ++i) {
//        for (int j = 0; j < image.rows; ++j) {
//            std::cout<<"image.at<double>(j,i)[0]"<<int(image.at<cv::Vec3d>(j,i)[0])<<std::endl;
//            std::cout<<"image.at<double>(j,i)[1]"<<int(image.at<cv::Vec3d>(j,i)[1])<<std::endl;
//            std::cout<<"image.at<double>(j,i)[2]"<<int(image.at<cv::Vec3d>(j,i)[2])<<std::endl;
//        }
//
//    }
//   std::vector<cv::Mat> image_split;
//   cv::split(image,image_split);
//   cv::Mat B = image_split[0];
//   cv::Mat G = image_split[1];
//   cv::Mat R = image_split[2];
//   cv::imshow("B",B);
//   cv::imshow("G",G);
//   cv::imshow("R",R);

    cv::Mat image_gray(90,90,CV_8UC1);
    randu(image_gray,cv::Scalar::all(0),cv::Scalar::all(255));
    std::cout<<"iamge_gray.channels:"<<image_gray.channels()<<std::endl;

    cv::namedWindow("iamge_gray",cv::WINDOW_NORMAL);
    cv::imshow("iamge_gray",image_gray);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}