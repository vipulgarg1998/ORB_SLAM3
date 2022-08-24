/**
 * @file rgbd_zed.cpp
 * @author Vipul Garg (garg.vipul7@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>
#include <sl/Camera.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);


cv::Mat slMat2cvMat(sl::Mat &input) {
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_file" << endl;
        return 1;
    }
    std::string svo_folder_dir = argv[3];
    
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);
    float imageScale = SLAM.GetImageScale();

    // Testing on 1 Cameras
    int num_cams = 1;
    sl::Camera zed;

    // ZED Camera Parameters
    sl::InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // Use a right-handed Y-up coordinate system
    init_parameters.depth_minimum_distance = 0.15;
    init_parameters.depth_stabilization = true;

    init_parameters.input.setFromSVOFile(svo_folder_dir.c_str());
    // Open Camera with the SVO file
    zed.open(init_parameters);

    // Configure spatial mapping parameters
    sl::SpatialMappingParameters mapping_parameters(sl::SpatialMappingParameters::MAPPING_RESOLUTION::LOW,
                                                    sl::SpatialMappingParameters::MAPPING_RANGE::AUTO);
    // In this cas we want to create a Mesh
    mapping_parameters.map_type = sl::SpatialMappingParameters::SPATIAL_MAP_TYPE::MESH;
    mapping_parameters.save_texture = true;
    // filter_params.set(sl::MeshFilterParameters::MESH_FILTER::LOW); // not available for fused point cloud

    sl::Mesh mesh; // Create a mesh object
    int timer=0;
    
    // Enable tracking and mapping
    zed.enablePositionalTracking();
    zed.enableSpatialMapping(mapping_parameters);

    // Read From the File
    int i = 0;
    for (int k = 0; k < num_cams; k++){

        auto camInfo = zed.getCameraInformation();
        auto camCalib = camInfo.camera_configuration.calibration_parameters.left_cam;
        auto fx = camCalib.fx;
        auto fy= camCalib.fy;
        auto cx = camCalib.cx;
        auto cy = camCalib.cy;
        auto width = camCalib.image_size.width;
        auto height = camCalib.image_size.height;
        auto disto = camCalib.disto;

        float temp_disto[12];
        std::copy(disto, disto + 12, temp_disto);
        std::vector<float> distCoeff;
        distCoeff.insert(distCoeff.begin(), std::begin(temp_disto), std::end(temp_disto));

        cout<<"Fx "<<fx<<" Fy "<<fy<<" Cx "<<cx<<" Cy "<<cy<<" Height "<<height<<" Width "<<width<<endl;
        for(int p = 0; p < 12; p++){
            cout<< " Disto: "<<disto[p]<<endl;
        }

        // Printing File Details
        std::cout<<"Reading File at "<<svo_folder_dir<<" With Serial Number "<<camInfo.serial_number<<std::endl;
        i++;
        if (i == num_cams){
            break;
        }
    }

    // ZED Images
    sl::Mat rgb_img, depth_img;
    // OpenCV Images
    cv::Mat cv_bgr_im, cv_depth_im;
    // Time Stamps
    vector<double> vTimestamps;

    while (true) {
        auto returned_state = zed.grab();
        if (returned_state == sl::ERROR_CODE::SUCCESS) {
            // std::cout<<"returned_state: "<<"Success"<<std::endl;
            // Retrieve Images from ZED
            zed.retrieveImage(rgb_img, sl::VIEW::LEFT, sl::MEM::CPU);
            zed.retrieveMeasure(depth_img, sl::MEASURE::DEPTH); // Retrieve depth
            // std::cout<<"Images Retrieved: "<<"Success"<<std::endl;

            // Convert ZED Image to OpenCV Image
            cv_depth_im = slMat2cvMat(depth_img);
            cv_bgr_im = slMat2cvMat(rgb_img);
            cv::cvtColor(cv_bgr_im, cv_bgr_im, cv::COLOR_BGRA2RGB);
            // std::cout<<"Images Converted: "<<"Success"<<std::endl;

            // Add timestamp
            double tframe = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getSeconds();
            // std::cout<<"Time Stamp: "<<tframe<<std::endl;

            if(imageScale != 1.f)
            {
                int width = cv_bgr_im.cols * imageScale;
                int height = cv_bgr_im.rows * imageScale;
                cv::resize(cv_bgr_im, cv_bgr_im, cv::Size(width, height));
                cv::resize(cv_depth_im, cv_depth_im, cv::Size(width, height));
            }
            // std::cout<<"Images Scaled: "<<"Success"<<std::endl;

            // std::cout<<"Processing Image"<<std::endl;
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            SLAM.TrackRGBD(cv_bgr_im,cv_depth_im,tframe);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            // Wait to load the next frame
            double T=1.0f/15;
            if(ttrack<T){
                usleep((T-ttrack)*1e6);
            }
        }
        else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            std::cout<<"returned_state: "<<"End of File"<<std::endl;
            break;
        }
        else {
            std::cout<<"returned_state: "<<"Something Else"<<std::endl;
            return 0;
        }
    }
    std::cout<<"Extract Map"<<std::endl;
    // Retrieve the spatial map
    zed.extractWholeSpatialMap(mesh);
    std::cout<<"Filter Map"<<std::endl;
    // Filter the mesh
    mesh.filter(sl::MeshFilterParameters::MESH_FILTER::LOW); // not available for fused point cloud
    std::cout<<"Apply Texture to Map"<<std::endl;
    // Apply the texture
    mesh.applyTexture(); // not available for fused point cloud
    std::cout<<"Save Map"<<std::endl;
    // Save the mesh in .obj format
    mesh.save("mesh.obj");



    // Stop all threads
    SLAM.Shutdown();
    return 0;
}
