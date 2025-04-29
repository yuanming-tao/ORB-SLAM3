#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>


inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

class YOLOv5Detector {
private:
cudaStream_t inference_stream_; 
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* mEngine;
    nvinfer1::IExecutionContext* context;
    std::thread mInferThread;
    bool mStop;
    std::queue<cv::Mat> mImgQueue;
    std::mutex mQueueMutex;
    std::condition_variable mCondVar;
    std::atomic<int> mActiveBuffer;
    struct Buffer {
        std::mutex mtx;
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::atomic<bool> data_ready{false};
        std::atomic<int> ref_count{0}; // 添加引用计数
    };
    Buffer mBuffer[2];

public:

    YOLOv5Detector();
    ~YOLOv5Detector();
    void Init(const std::string& enginePath);
    void AsyncDetect(cv::Mat& img, std::vector<cv::Rect>& boxes, std::vector<int>& class_ids);
    cv::Mat SyncDetect(cv::Mat& img, 
        std::vector<cv::Rect>& boxes, 
        std::vector<int>& class_ids) ;
    
        void *d_input = nullptr; 
    void *d_output = nullptr;

    // Letterbox参数
    float scale_ratio = 1.0f;
    int pad_w = 0, pad_h = 0;

    // 锚点参数（根据模型配置）
    const std::vector<std::vector<float>> anchors = {
        // P3/8层（检测小目标）的三个锚点
        {10,13}, {16,30}, {33,23},
        
        // P4/16层（检测中目标）的三个锚点  
        {30,61}, {62,45}, {59,119},
        
        // P5/32层（检测大目标）的三个锚点
        {116,90}, {156,198}, {373,326}
    }; 

    // 双缓冲互斥锁
    std::mutex mBufferSwitchMtx;
private:
    void InferenceThread();
    void PostProcess(float* output, cv::Mat& img, std::vector<cv::Rect>& boxes, std::vector<int>& class_ids);
};