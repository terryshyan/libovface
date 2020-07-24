// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "recognizer.hpp"

#include <algorithm>
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;
using namespace ovface;

FaceRecognizerDefault::FaceRecognizerDefault(
  const CnnConfig& landmarks_detector_config,
  const CnnConfig& reid_config,
  const DetectorConfig& face_registration_det_config,
  const std::string& face_gallery_path,
  double reid_threshold,
  DistaceAlgorithm reid_algorithm,
  int min_size_fr,
  bool crop_gallery,
  bool greedy_reid_matching
)
  : landmarks_detector(landmarks_detector_config),
    face_reid(reid_config),
    face_gallery(face_gallery_path, reid_threshold, reid_algorithm, min_size_fr, crop_gallery,
                 face_registration_det_config, landmarks_detector, face_reid,
                 greedy_reid_matching) {
  if (face_gallery.size() == 0) {
    std::cout << "Face reid gallery is empty!" << std::endl;
  } else {
    std::cout << "Face reid gallery size: " << face_gallery.size() << std::endl;
  }
}

bool FaceRecognizerDefault::LabelExists(const std::string &label) const {
  return face_gallery.LabelExists(label);
}

std::string FaceRecognizerDefault::GetLabelByID(int id) const {
  return face_gallery.GetLabelByID(id);
}

int FaceRecognizerDefault::GetIDByID(int id) const {
  return face_gallery.GetIDByID(id);
}

std::vector<std::string> FaceRecognizerDefault::GetIDToLabelMap() const {
  return face_gallery.GetIDToLabelMap();
}

std::vector<int> FaceRecognizerDefault::Recognize(const cv::Mat& frame, const DetectedObjects& faces) {
  std::vector<cv::Mat> face_rois;
  std::vector<cv::Rect> face_rects;
  // transfer rectangle to square as recognize input is square
  for (const auto& face : faces) {
    cv::Rect rect = face.rect;
    RectangletoSquare(rect, frame);
    face_rois.push_back(frame(rect));
    face_rects.push_back(rect);
  }
  
  std::vector<cv::Mat> landmarks, embeddings;
  landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
  //AlignFaces(&face_rois, &landmarks);
  AlignFaces2(&face_rois, &landmarks, frame, face_rects);
  face_reid.Compute(face_rois, &embeddings);
  return face_gallery.GetIDsByEmbeddings(embeddings);
}

std::vector<cv::Mat> FaceRecognizerDefault::Recognize2(const cv::Mat& frame, const DetectedObjects& faces) {
  std::vector<cv::Mat> face_rois;
  std::vector<cv::Rect> face_rects;
  // transfer rectangle to square as recognize input is square
  for (const auto& face : faces) {
    cv::Rect rect = face.rect;
    RectangletoSquare(rect, frame);
    face_rois.push_back(frame(rect));
    face_rects.push_back(rect);
  }
  
  std::vector<cv::Mat> landmarks, embeddings;
  landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
  //AlignFaces(&face_rois, &landmarks);
  AlignFaces2(&face_rois, &landmarks, frame, face_rects);
  face_reid.Compute(face_rois, &embeddings);
  return embeddings;
}

void FaceRecognizerDefault::PrintPerformanceCounts(
  const std::string &landmarks_device, const std::string &reid_device) {
  landmarks_detector.PrintPerformanceCounts(landmarks_device);
  face_reid.PrintPerformanceCounts(reid_device);
}

void FaceRecognizerDefault::updateIdentityDB(const std::vector<CIdentityParams> &params) {
  face_gallery.updateIdentityDB(params);
}


FaceRecognizerLfw::FaceRecognizerLfw(
  const CnnConfig& landmarks_detector_config,
  const CnnConfig& reid_config
)
  : landmarks_detector(landmarks_detector_config),
  face_reid(reid_config)
{

}

std::vector<cv::Mat> FaceRecognizerLfw::Recognize(const cv::Mat& frame, const DetectedObjects& faces) {
  std::vector<cv::Mat> face_rois;
  std::vector<cv::Rect> face_rects;
  // transfer rectangle to square as recognize input is square
  for (const auto& face : faces) {
    cv::Rect rect = face.rect;
    RectangletoSquare(rect, frame);
    face_rois.push_back(frame(rect));
    face_rects.push_back(rect);
  }
  
  std::vector<cv::Mat> landmarks, embeddings;
  landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
  //AlignFaces(&face_rois, &landmarks);
  AlignFaces2(&face_rois, &landmarks, frame, face_rects);
  face_reid.Compute(face_rois, &embeddings);
  return embeddings;
}

