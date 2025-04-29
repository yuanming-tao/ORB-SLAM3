#include "YOLOv5Detector.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};
static Logger gLogger;
struct Buffer
{
    std::mutex mtx;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::atomic<bool> data_ready{false};
    std::atomic<int> ref_count{0}; // 添加引用计数
};

#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t status = call;                                           \
        if (status != cudaSuccess)                                           \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

YOLOv5Detector::YOLOv5Detector() : runtime(nullptr), mEngine(nullptr), context(nullptr), mStop(false), mActiveBuffer(0)
{
    cudaStreamCreate(&inference_stream_); 
    for (int i = 0; i < 2; ++i)
    {
        mBuffer[i].data_ready.store(false);
    }
}

YOLOv5Detector::~YOLOv5Detector()
{
    cudaStreamDestroy(inference_stream_); 
    mStop = true;
    mCondVar.notify_all();
    if (mInferThread.joinable())
    {
        mInferThread.join();
    }
    if (context)
        delete context;
    if (mEngine)
        delete mEngine;
    if (runtime)
        delete runtime;
}

void YOLOv5Detector::Init(const std::string &enginePath)
{
    try
    {
        runtime = nvinfer1::createInferRuntime(gLogger);
        std::ifstream engineFile(enginePath, std::ios::binary);
        std::vector<char> engineData(
            (std::istreambuf_iterator<char>(engineFile)),
            std::istreambuf_iterator<char>());

        mEngine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        context = mEngine->createExecutionContext();

        cudaMalloc(&d_input, 3 * 640 * 640 * sizeof(float));
        cudaMalloc(&d_output, 25200 * 85 * sizeof(float));

        // 获取输入输出索引
        int inputIndex = -1, outputIndex = -1;
        for (int i = 0; i < mEngine->getNbIOTensors(); ++i)
        {
            const char *name = mEngine->getIOTensorName(i);
            //     std::cout << "Tensor Name: " << name << std::endl;
            nvinfer1::TensorIOMode ioMode = mEngine->getTensorIOMode(name);

            if (ioMode == nvinfer1::TensorIOMode::kINPUT)
            {
                if (strcmp(name, "images") == 0)
                    inputIndex = i;
            }
            else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT)
            {
                if (strcmp(name, "output0") == 0)
                    outputIndex = i;
            }
        }

        if (inputIndex == -1 || outputIndex == -1)
        {
            throw std::runtime_error("无法找到输入或输出张量");
        }
        context->setTensorAddress("images", d_input);
        context->setTensorAddress("output0", d_output);
        mInferThread = std::thread(&YOLOv5Detector::InferenceThread, this);
    }
    catch (const std::exception &e)
    {
        std::cerr << "初始化失败: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void YOLOv5Detector::InferenceThread()
{
    try
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        while (!mStop)
        {
            cv::Mat img;
            {
                std::unique_lock<std::mutex> lock(mQueueMutex);
                mCondVar.wait(lock, [&]
                              { return !mImgQueue.empty() || mStop; });
                if (mStop)
                    break;
                img = std::move(mImgQueue.front());
                mImgQueue.pop();
            }

            int img_h = img.rows, img_w = img.cols;
            float scale = std::min(640.0f / img_w, 640.0f / img_h);
            int new_w = img_w * scale, new_h = img_h * scale;
            int pad_w = (640 - new_w) / 2, pad_h = (640 - new_h) / 2;

            cv::Mat resized;
            cv::resize(img, resized, cv::Size(new_w, new_h));
            cv::copyMakeBorder(resized, resized, pad_h, pad_h, pad_w, pad_w,
                               cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

            CHECK_CUDA(cudaMemcpyAsync(d_input, resized.ptr<float>(), 3 * 640 * 640 * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));

            if (!context->enqueueV3(stream))
            {
                throw std::runtime_error("推理执行失败");
            }

            float *output = new float[25200 * 85];
            CHECK_CUDA(cudaMemcpyAsync(output, d_output, 25200 * 85 * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);

            std::vector<cv::Rect> tmpBoxes;
            std::vector<float> confidences;
            std::vector<int> tmpClassIds; // 添加临时类别ID存储

            for (int i = 0; i < 25200; ++i)
            {
                float *det = &output[i * 85];
                float obj_conf = det[4];
                float cls_conf = det[5];
                float total_conf = obj_conf * cls_conf;

                if (total_conf > 0.02)
                {
                    int anchor_idx = (i / (25200 / 3)) % 3;
                    float x_center = (sigmoid(det[0]) * 2 - 0.5 + (i % 3) * 0.333) * 640;
                    float y_center = (sigmoid(det[1]) * 2 - 0.5 + (i / 3 % 3) * 0.333) * 640;
                    float width = pow(sigmoid(det[2]) * 2, 2) * anchors[anchor_idx][0];
                    float height = pow(sigmoid(det[3]) * 2, 2) * anchors[anchor_idx][1];

                    x_center = (x_center - pad_w) / scale;
                    y_center = (y_center - pad_h) / scale;
                    width /= scale;
                    height /= scale;

                    tmpBoxes.emplace_back(x_center - width / 2, y_center - height / 2, width, height);
                    confidences.push_back(total_conf); // 使用总置信度
                    tmpClassIds.push_back(0);          // 假设我们只检测一种类别
                }
            }

            // std::cout<<"tmpBoxes.size():"<<tmpBoxes.size()<<std::endl;
            std::vector<int> indices;
            cv::dnn::NMSBoxes(tmpBoxes, confidences, 0.2, 0.45, indices);

            // 准备最终结果
            std::vector<cv::Rect> finalBoxes;
            std::vector<int> finalClassIds;

            for (int idx : indices)
            {
                finalBoxes.push_back(tmpBoxes[idx]);
                finalClassIds.push_back(tmpClassIds[idx]);
            }
            std::cout << "finalBoxes.size():" << finalBoxes.size() << std::endl;
            // 更新缓冲区
            int write_buffer = mActiveBuffer.load(std::memory_order_acquire);
            {
                std::unique_lock<std::mutex> lock(mBuffer[write_buffer].mtx);
                while (mBuffer[write_buffer].ref_count > 0)
                {
                    std::this_thread::yield();
                }
                mBuffer[write_buffer].boxes = finalBoxes;
                mBuffer[write_buffer].class_ids = finalClassIds; // 保存类别ID
                mBuffer[write_buffer].data_ready.store(true, std::memory_order_release);

                // 切换缓冲区
                mActiveBuffer.store(1 - write_buffer, std::memory_order_release);
            }

            delete[] output;
        }

        cudaStreamDestroy(stream);
    }
    catch (const std::exception &e)
    {
        std::cerr << "推理线程异常: " << e.what() << std::endl;
    }
}
void YOLOv5Detector::PostProcess(float *output, cv::Mat &img,
                                 std::vector<cv::Rect> &boxes, std::vector<int> &class_ids)
{
    std::vector<cv::Rect> tmpBoxes;
    std::vector<float> confidences;
    std::vector<int> tmpClassIds;

    for (int i = 0; i < 25200; ++i)
    {
        float *det = &output[i * 85];
        float conf = det[4];
        //        std::cout<<"det[4]:"<<det[4]<<std::endl;
        if (conf > 0.25)
        {
            float x = det[0] * img.cols / 640;
            float y = det[1] * img.rows / 640;
            float w = det[2] * img.cols / 640;
            float h = det[3] * img.rows / 640;

            tmpBoxes.emplace_back(x - w / 2, y - h / 2, w, h);
            confidences.push_back(conf);
            //       std::cout<<"success"<<std::endl;
            // 这里假设类别索引从5开始
            // tmpClassIds.push_back(std::max_element(det + 5, det + 85) - (det + 5));
        }
    }

    std::vector<int> indices;
    //    std::cout<<"tmpBoxes.size():"<<tmpBoxes.size()<<std::endl;
    cv::dnn::NMSBoxes(tmpBoxes, confidences, 0.5, 0.45, indices);

    for (int idx : indices)
    {
        boxes.push_back(tmpBoxes[idx]);
        class_ids.push_back(tmpClassIds[idx]);
    }
}

void YOLOv5Detector::AsyncDetect(cv::Mat &img,
                                 std::vector<cv::Rect> &boxes,
                                 std::vector<int> &class_ids)
{
    // 将图像添加到队列
    {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        if (mImgQueue.size() < 5)
        {
            mImgQueue.emplace(img.clone());
            mCondVar.notify_one(); // 确保通知推理线程
        }
    }

    // 获取非活动缓冲区索引（这是已处理完成的缓冲区）
    const int ready_buffer = 1 - mActiveBuffer.load(std::memory_order_acquire);
    auto &buf = mBuffer[ready_buffer];

    // 尝试锁定并读取结果
    std::unique_lock<std::mutex> lock(buf.mtx, std::try_to_lock);
    if (lock.owns_lock() && buf.data_ready.load(std::memory_order_acquire))
    {
        buf.ref_count.fetch_add(1);
        boxes = buf.boxes;
        class_ids = buf.class_ids; // 正确获取类别ID
        buf.ref_count.fetch_sub(1);
    }
    else
    {
        // 如果无法获取结果，返回空结果
        boxes.clear();
        class_ids.clear(); // 同时清空类别ID
    }
}

cv::Mat YOLOv5Detector::SyncDetect(cv::Mat &img,
                                std::vector<cv::Rect> &boxes,
                                std::vector<int> &class_ids)
{
    // 预处理参数
    const float conf_threshold = 0.2;
    const float iou_threshold = 0.45;
    const int input_size = 640;

    // 图像预处理
    cv::Mat resized;
    float scale = std::min(input_size / (float)img.cols, input_size / (float)img.rows);
    cv::resize(img, resized, cv::Size(img.cols * scale, img.rows * scale));
    cv::Mat padded(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
    //cv::imwrite("padded.jpg", padded);
    resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));

    // 转换为float并归一化
    cv::Mat blob;
    padded.convertTo(blob, CV_32FC3, 1.0 / 255.0);

    // 转换为CHW格式
    std::vector<cv::Mat> channels(3);
    cv::split(blob, channels);
    std::vector<float> input_data;
    for (auto &c : channels)
    {
        input_data.insert(input_data.end(), (float *)c.data, (float *)c.data + input_size * input_size);
    }

    // 拷贝到GPU
    cudaMemcpyAsync(d_input, input_data.data(), 3*input_size*input_size*sizeof(float),
    cudaMemcpyHostToDevice, inference_stream_);  

    // 同步推理
    context->enqueueV3(inference_stream_); 

    // 获取输出
    float *output = new float[25200 * 85];
    cudaMemcpyAsync(output, d_output, 25200 * 85*sizeof(float),
    cudaMemcpyDeviceToHost, inference_stream_); 
    cudaStreamSynchronize(inference_stream_); 

    // 解析检测结果
    std::vector<cv::Rect> proposals;
    std::vector<float> confidences;
    std::vector<int> class_list;

    for (int i = 0; i < 25200; ++i)
    {
        float *det = &output[i * 85];
        float confidence = det[4];
        if (confidence < conf_threshold)
            continue;

        // 获取类别
        float *classes = det + 5;
        int class_id = std::max_element(classes, classes + 80) - classes;
        if (class_id != 0)
            continue;
        float class_score = classes[class_id];
        if (class_score < conf_threshold)
            continue;

        // 计算实际坐标
        float xc = det[0] / scale;
        float yc = det[1] / scale;
        float w = det[2] / scale;
        float h = det[3] / scale;

        int x = xc - w / 2;
        int y = yc - h / 2;
        proposals.emplace_back(x, y, w, h);
        confidences.push_back(confidence * class_score);
        class_list.push_back(class_id);
    }
    delete[] output;

    // NMS处理
    std::vector<int> indices;
    cv::dnn::NMSBoxes(proposals, confidences, conf_threshold, iou_threshold, indices);

    // 输出结果
    for (int idx : indices)
    {
        boxes.push_back(proposals[idx]);
        class_ids.push_back(class_list[idx]);
    }

    cv::Mat mask = cv::Mat::ones(img.size(), CV_8UC1) * 255; // 初始化为全白
    for (const auto &box : boxes) {
        cv::rectangle(mask, box, cv::Scalar(0), -1); // 绘制黑色矩形
    }
    return mask;
}