#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "ovface.h"


using namespace ovface;

void DrawObject(cv::Mat frame, cv::Rect rect, const std::string& label) {
  const cv::Scalar text_color(0, 0, 255);
  const cv::Scalar bbox_color(255, 255, 255);
  bool plot_bg = true;
  
  cv::rectangle(frame, rect, bbox_color);

  if (plot_bg && !label.empty()) {
    int baseLine = 0;
    const cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_size.height), cv::Point(rect.x + label_size.width, rect.y + baseLine), bbox_color, cv::FILLED);
  }
  
  if (!label.empty()) {
    cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1, text_color, 1, cv::LINE_AA);
  }
}

void testLib() {
  const char *src = "./share/test.mp4";
  const char *dst = "./share/out_v.mp4";
  
  //----Create chan
  CVAChanParams param;
  param.device = "CPU";
  param.faceDetectModelPath = "./models/face-detection-adas-0001.xml";
  param.landmarksModelPath = "./models/landmarks-regression-retail-0009.xml";
  param.faceRecogModelPath = "./models/model-y1-0000.xml";
  param.reidGalleryPath = "./share/faces_gallery.json";
  param.detectThreshold = 0.6;
  param.reidThreshold = 0.57;
  param.maxBatchSize = 16;
  param.detectInterval = 0;
    
  VAChannel * chan =VAChannel::create(param);

  //----Capture
  cv::VideoCapture capture;
  capture.open(src);
  if (!capture.isOpened()) {
    std::cout << "[ERROR] Fail to capture video." << std::endl;
    return;
  }
  int w=capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int h=capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  
  cv::VideoWriter writer;
  writer.open(dst,cv::VideoWriter::fourcc('X','2','6','4'),30,cv::Size(w,h));

  while(true) {
    cv::Mat frame;
    capture >> frame;
    
    if (frame.empty()) {
      std::cout << " Finished" << std::endl;
      break;
    }
    cv::Mat frame2 = frame.clone();
    CFrameData frameData;
    frameData.format = FRAME_FOMAT_BGR;
    frameData.pFrame = frame.data;
    frameData.width = w;
    frameData.height = h;
    std::vector<CResult> results;
    chan->process(frameData, results);
    
    //std::cout << results.size() << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
      CResult result = results[i];
      cv::Rect rect;
      rect.x = result.rect.left;
      rect.y = result.rect.top;
      rect.height = result.rect.bottom - result.rect.top;
      rect.width = result.rect.right - result.rect.left;
      DrawObject(frame2, rect, result.label);
      //std::cout << result.frameId << "/" << i << " rect=" << result.rect.left << "," << result.rect.top << " " << result.rect.right << ":" << result.rect.bottom << "   " << result.label << std::endl;
    }
    writer.write(frame2);
  }
  
  VAChannel::destroyed(chan);
  writer.release();
}

int main(int argc, char* argv[]) {
    try {
      testLib();
    }
    catch (const std::exception& error) {
        std::cout << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cout << "Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "Execution successful" << std::endl;

    return 0;
}

