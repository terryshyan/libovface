// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detector.hpp"

#include <algorithm>
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <inference_engine.hpp>

#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;

#define SSD_EMPTY_DETECTIONS_INDICATOR -1.0

using namespace ovface;

namespace {
cv::Rect TruncateToValidRect(const cv::Rect& rect,
                             const cv::Size& size) {
  auto tl = rect.tl(), br = rect.br();
  tl.x = std::max(0, std::min(size.width - 1, tl.x));
  tl.y = std::max(0, std::min(size.height - 1, tl.y));
  br.x = std::max(0, std::min(size.width, br.x));
  br.y = std::max(0, std::min(size.height, br.y));
  int w = std::max(0, br.x - tl.x);
  int h = std::max(0, br.y - tl.y);
  return cv::Rect(tl.x, tl.y, w, h);
}

cv::Rect IncreaseRect(const cv::Rect& r, float coeff_x,
                      float coeff_y)  {
  cv::Point2f tl = r.tl();
  cv::Point2f br = r.br();
  cv::Point2f c = (tl * 0.5f) + (br * 0.5f);
  cv::Point2f diff = c - tl;
  cv::Point2f new_diff{diff.x * coeff_x, diff.y * coeff_y};
  cv::Point2f new_tl = c - new_diff;
  cv::Point2f new_br = c + new_diff;
  cv::Point new_tl_int {static_cast<int>(std::floor(new_tl.x)), static_cast<int>(std::floor(new_tl.y))};
  cv::Point new_br_int {static_cast<int>(std::ceil(new_br.x)), static_cast<int>(std::ceil(new_br.y))};
  return cv::Rect(new_tl_int, new_br_int);
}

cv::Rect RectToSquare(const cv::Rect& r){
  cv::Point2f tl = r.tl();
  cv::Point2f br = r.br();
  cv::Point2f c = (tl * 0.5f) + (br * 0.5f);
  
  float max = r.width;
  if (max < r.height)
    max = r.height;
  cv::Point2f new_diff{ max/2.0f, max/2.0f };
  cv::Point2f new_tl = c - new_diff;
  cv::Point2f new_br = c + new_diff;
  cv::Point new_tl_int{ static_cast<int>(std::floor(new_tl.x)), static_cast<int>(std::floor(new_tl.y)) };
  cv::Point new_br_int{ static_cast<int>(std::ceil(new_br.x)), static_cast<int>(std::ceil(new_br.y)) };
  return cv::Rect(new_tl_int, new_br_int);
}
}  // namespace

void FaceDetection::submitRequest() {
  if (!enqueued_frames_) return;
  enqueued_frames_ = 0;
  BaseCnnDetection::submitRequest();
}

void FaceDetection::enqueue(const cv::Mat &frame) {
  if (!request) {
    request = net_.CreateInferRequestPtr();
  }

  width_ = static_cast<float>(frame.cols);
  height_ = static_cast<float>(frame.rows);

  Blob::Ptr inputBlob = request->GetBlob(input_name_);

  matU8ToBlob<uint8_t>(frame, inputBlob);

  enqueued_frames_ = 1;
}

FaceDetection::FaceDetection(const DetectorConfig& config) :
  BaseCnnDetection(config.is_async), config_(config) {

  //std::cout << "Loading network files for Face Detection" << std::endl;
  topoName = "face detector";
  auto cnnNetwork = config.ie.ReadNetwork(config.path_to_model);

  // ---------------------------Check inputs -------------------------------------------------------------
  //std::cout << "Checking Face Detection network inputs" << std::endl;
  InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
  if (inputInfo.size() != 1) {
    THROW_IE_EXCEPTION << "Face Detection network should have only one input";
  }
  InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
  inputInfoFirst->setPrecision(Precision::U8);
  //inputInfoFirst->getInputData()->setLayout(Layout::NCHW);

  const SizeVector inputDims = inputInfoFirst->getTensorDesc().getDims();
  network_input_height_ = inputDims[2];
  network_input_width_ = inputDims[3];

  // ---------------------------Check outputs ------------------------------------------------------------
  //std::cout << "Checking Face Detection network outputs" << std::endl;
  OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
  if (outputInfo.size() == 1) {
    DataPtr& _output = outputInfo.begin()->second;
    output_name_ = outputInfo.begin()->first;
    const SizeVector outputDims = _output->getTensorDesc().getDims();
    max_detections_count_ = outputDims[2];
    if (config_.max_detections_count && config_.max_detections_count < max_detections_count_) {
      max_detections_count_ = config_.max_detections_count;
    }

    object_size_ = outputDims[3];
    if (object_size_ != 7) {
      throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
      throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
        std::to_string(outputDims.size()));
    }
    _output->setPrecision(Precision::FP32);
  }
  else {
    for (const auto& outputLayer : outputInfo) {
      const SizeVector outputDims = outputLayer.second->getTensorDesc().getDims();
      if (outputDims.size() == 2 && outputDims.back() == 5) {
        output_name_ = outputLayer.first;
        max_detections_count_ = outputDims[0];
        if (config_.max_detections_count && config_.max_detections_count < max_detections_count_) {
          max_detections_count_ = config_.max_detections_count;
        }
        object_size_ = outputDims.back();
        outputLayer.second->setPrecision(Precision::FP32);
      }
      else if (outputDims.size() == 1 && outputLayer.second->getPrecision() == Precision::I32) {
        labels_output_ = outputLayer.first;
      }
    }
    if (output_name_.empty() || labels_output_.empty()) {
      throw std::logic_error("Face Detection network must contain ether single DetectionOutput or "
        "'boxes' [nx5] and 'labels' [n] at least, where 'n' is a number of detected objects.");
    }
  }
  input_name_ = inputInfo.begin()->first;
  
  if (config.networkCfg.nCpuThreadsNum > 0) {
    std::map<std::string, std::string> loadParams;
    loadParams[PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(config.networkCfg.nCpuThreadsNum);
    loadParams[PluginConfigParams::KEY_CPU_BIND_THREAD] = config.networkCfg.bCpuBindThread ? PluginConfigParams::YES : PluginConfigParams::NO;
    loadParams[PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(config.networkCfg.nCpuThroughputStreams);
    net_ = config_.ie.LoadNetwork(cnnNetwork, config_.deviceName, loadParams);
  } else {
    net_ = config_.ie.LoadNetwork(cnnNetwork, config_.deviceName);
  }
}

DetectedObjects FaceDetection::fetchResults() {
  DetectedObjects results;
  const float* detections = request->GetBlob(output_name_)->buffer().as<float*>();
  
	for (int i = 0; i < max_detections_count_ && object_size_ == 5; i++) {
		DetectedObject r;
		r.confidence = detections[i * object_size_ + 4];
		if (r.confidence <= config_.confidence_threshold) {      
			continue;
		}
    //std::cout << "confidence: " << r.confidence << std::endl;


		r.rect.x = static_cast<int>(detections[i * object_size_ + 0] / network_input_width_ * width_);
		r.rect.y = static_cast<int>(detections[i * object_size_ + 1] / network_input_height_ * height_);
		r.rect.width = static_cast<int>(detections[i * object_size_ + 2] / network_input_width_ * width_ - r.rect.x);
		r.rect.height = static_cast<int>(detections[i * object_size_ + 3] / network_input_height_ * height_ - r.rect.y);

		// Make square and enlarge face bounding box for more robust operation of face analytics networks
		int bb_width = r.rect.width;
		int bb_height = r.rect.height;

		int bb_center_x = r.rect.x + bb_width / 2;
		int bb_center_y = r.rect.y + bb_height / 2;

		int max_of_sizes = std::max(bb_width, bb_height);

    int bb_new_length;
    float scalex = r.rect.width / width_;
    float scaley = r.rect.height / height_;
    float scale = std::min(scalex, scaley);
    if (scale < 0.85)
      bb_new_length = static_cast<int>(config_.increase_scale * max_of_sizes);
    else
      bb_new_length = max_of_sizes;

    int bb_new_half = bb_new_length / 2;
    bb_new_half = std::min(bb_new_half, bb_center_x);
    bb_new_half = std::min(bb_new_half, bb_center_y);
    bb_new_half = std::min(bb_new_half, static_cast<int>(width_)- bb_center_x);
    bb_new_half = std::min(bb_new_half, static_cast<int>(height_) - bb_center_y);
    bb_new_length = bb_new_half * 2;
    
		r.rect.x = bb_center_x - bb_new_half;
    r.rect.y = bb_center_y - bb_new_half;

		r.rect.width = bb_new_length;
		r.rect.height = bb_new_length;

    std::cout << "image: " << width_ << "x" << height_ << std::endl;
    std::cout << "rect: " << r.rect.x << "," << r.rect.y << " " << r.rect.width << "x" << r.rect.height << std::endl;

    if (r.rect.area() > 0) {
      results.push_back(r);
    }
	}

  for (int i = 0; i < max_detections_count_ && object_size_ == 7; i++) {
    float image_id = detections[i * object_size_ + 0];
    if (image_id < 0) {
      break;
    }
    DetectedObject r;
    r.confidence = detections[i * object_size_ + 2];

    if (r.confidence <= config_.confidence_threshold) {
      continue;
    }
    //std::cout << "confidence: " << r.confidence << std::endl;

    r.rect.x = static_cast<int>(detections[i * object_size_ + 3] * width_);
    r.rect.y = static_cast<int>(detections[i * object_size_ + 4] * height_);
    r.rect.width = static_cast<int>(detections[i * object_size_ + 5] * width_ - r.rect.x);
    r.rect.height = static_cast<int>(detections[i * object_size_ + 6] * height_ - r.rect.y);

    // Make square and enlarge face bounding box for more robust operation of face analytics networks
    int bb_width = r.rect.width;
    int bb_height = r.rect.height;

    int bb_center_x = r.rect.x + bb_width / 2;
    int bb_center_y = r.rect.y + bb_height / 2;

    int max_of_sizes = std::max(bb_width, bb_height);

    int bb_new_length = static_cast<int>(config_.increase_scale * max_of_sizes);

    int bb_new_half = bb_new_length / 2;
    bb_new_half = std::min(bb_new_half, bb_center_x);
    bb_new_half = std::min(bb_new_half, bb_center_y);
    bb_new_half = std::min(bb_new_half, static_cast<int>(width_) - bb_center_x);
    bb_new_half = std::min(bb_new_half, static_cast<int>(height_) - bb_center_y);
    bb_new_length = bb_new_half * 2;

    r.rect.x = bb_center_x - bb_new_half;
    r.rect.y = bb_center_y - bb_new_half;

    r.rect.width = bb_new_length;
    r.rect.height = bb_new_length;

    if (r.rect.area() > 0) {
      results.push_back(r);
    }
  }
  return results;
}
