/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include<opencv2/core/core.hpp>
#include <sys/stat.h>
#include<System.h>
#include <limits>


using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

string get_formatted_datetime();
void appendToCSV(const std::string& filename, const std::vector<std::string>& rowData);
string getFileName(const std::string& path);
template<typename T>
std::string floatToExactString(T value);

int main(int argc, char **argv)
{
    cout<<"run"<<endl;
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);
        
       // while(k) {std::cout<<imRGB<<std::endl;k=0;}
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    
////////////////////////////////////////////////////////////////////////////////////////////
    mode_t permissions = 0777; 
    std::string absolute_path = "/home/wenkai/实验记录";
    std::string datetime = get_formatted_datetime();
    std::string full_path = absolute_path + "/" + datetime;
    mkdir(full_path.c_str(), permissions);
    std::string file_path = full_path + "/CameraTrajectory.txt";
    std::string file_path2 = full_path + "/KeyFrameTrajectory.txt";
    
    // Save camera trajectory
    SLAM.SaveTrajectoryTUM(file_path);
    SLAM.SaveKeyFrameTrajectoryTUM(file_path2);
    
    std::string version = "原始版本";
    std::string remark = "yolo测试版";
    std::string file_path3 ="/home/wenkai/实验记录/data.csv";
    std::string file_name = getFileName(argv[3]);
    appendToCSV(file_path3, {file_name, datetime, floatToExactString(vTimesTrack[nImages/2]), floatToExactString(totaltime/nImages),floatToExactString(totaltime), version, remark });

  //////////////////////////////////////////////////////////////////////////////////////////////  
    ofstream f("track_time.log");
   for(auto t : vTimesTrack)
     f << t << endl;
      f.close();
    

    

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}


// 生成格式化的日期时间字符串
std::string get_formatted_datetime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);

    std::stringstream ss;
    ss << std::put_time(local_time, "%Y-%m-%d %H:%M");
    return ss.str();
}


// 将数据追加到 CSV 文件
void appendToCSV(const std::string& filename, const std::vector<std::string>& rowData) {
    std::ofstream file(filename, std::ios::app);  // 追加模式打开
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    // 转义特殊字符
    for (size_t i = 0; i < rowData.size(); ++i) {
        if (rowData[i].find('"') != std::string::npos || 
            rowData[i].find(',') != std::string::npos) {
            file << "\"" << rowData[i] << "\"";
        } else {
            file << rowData[i];
        }
        
        if (i != rowData.size() - 1) {
            file << ",";
        }
    }
    
    file << "\n";
    file.close();
}

std::string getFileName(const std::string& path) {
    // 同时处理Windows和Linux路径分隔符
    size_t pos = path.find_last_of("/\\");
    
    // 如果找到分隔符，返回分隔符后的内容
    if (pos != std::string::npos) {
        // 排除以分隔符结尾的情况（如/path/to/folder/）
        if (pos == path.length() - 1) {
            return "";
        }
        return path.substr(pos + 1);
    }
 // 没有分隔符直接返回原字符串
    return path;
}


template<typename T>
std::string floatToExactString(T value) {
    static_assert(std::is_floating_point<T>::value, "仅限浮点类型");
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<T>::max_digits10)
        << value;
    return oss.str();
}
