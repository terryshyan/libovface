// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "face_reid.hpp"
#include "tracker.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

#include <opencv2/opencv.hpp>
#include "ovface.h"

using namespace ovface;

namespace {
float ComputeReidDistance(const cv::Mat& descr1, const cv::Mat& descr2, DistaceAlgorithm algorithm) {
  if (algorithm == DISTANCE_EUCLIDEAN) {
    float dist = 0.0f;
    cv::Mat col_mean_src;
    reduce(descr1, col_mean_src, 0, cv::REDUCE_AVG);
    for (int i = 0; i < descr1.rows; i++) {
      descr1.row(i) -= col_mean_src;
    }

    cv::Mat col_mean_dst;
    reduce(descr2, col_mean_dst, 0, cv::REDUCE_AVG);
    for (int i = 0; i < descr2.rows; i++) {
      descr2.row(i) -= col_mean_dst;
    }

    cv::Scalar mean, dev_src, dev_dst;
    cv::Mat feature1(descr1);
    cv::meanStdDev(descr1, mean, dev_src);
    dev_src(0) = std::max(static_cast<double>(std::numeric_limits<float>::epsilon()), dev_src(0));
    feature1 /= dev_src(0);
    cv::normalize(feature1, descr1);

    cv::Mat feature2(descr2);
    cv::meanStdDev(descr2, mean, dev_dst);
    dev_dst(0) = std::max(static_cast<double>(std::numeric_limits<float>::epsilon()), dev_dst(0));
    feature2 /= dev_dst(0);
    cv::normalize(feature2, descr2);
    cv::Mat diff = descr2 - descr1;
    dist = sqrt(diff.dot(diff));
    return dist;
  } else {
    float xy = static_cast<float>(descr1.dot(descr2));
    float xx = static_cast<float>(descr1.dot(descr1));
    float yy = static_cast<float>(descr2.dot(descr2));
    float norm = sqrt(xx) * sqrt(yy) + 1e-6f;
    return 1.0f - xy / norm;
  }
}

bool file_exists(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

inline char separator() {
#ifdef _WIN32
  return '\\';
#else
  return '/';
#endif
}

std::string folder_name(const std::string& path) {
  size_t found_pos;
  found_pos = path.find_last_of(separator());
  if (found_pos != std::string::npos)
    return path.substr(0, found_pos);
  return std::string(".") + separator();
}

}  // namespace

const char EmbeddingsGallery::unknown_label[] = "Unknown";
const int EmbeddingsGallery::unknown_id = TrackedObject::UNKNOWN_LABEL_IDX;

RegistrationStatus EmbeddingsGallery::RegisterIdentity(const std::string& identity_label,
    const cv::Mat& image,
    int min_size_fr, bool crop_gallery,
    FaceDetection& detector,
    const VectorCNN& landmarks_det,
    const VectorCNN& image_reid,
    cv::Mat& embedding) {
  cv::Mat target = image;
  cv::Rect rect(0, 0, image.cols, image.rows);
  if (crop_gallery) {
    detector.enqueue(image);
    detector.submitRequest();
    detector.wait();
    DetectedObjects faces = detector.fetchResults();
    if (faces.size() == 0) {
      return RegistrationStatus::FAILURE_NOT_DETECTED;
    }
    
    DetectedObject face0 = faces[0];
    for (size_t i = 1; i < faces.size(); i++) {
      DetectedObject face = faces[i];
      if (face.rect.area() > face0.rect.area())
        face0 = face;
    }
    rect = face0.rect;
    // transfer rectangle to square as recognize input is square
    RectangletoSquare(rect, image);
    cv::Mat face_roi = image(rect);
    target = face_roi;
  }
  
  if ((target.rows < min_size_fr) && (target.cols < min_size_fr)) {
    return RegistrationStatus::FAILURE_LOW_QUALITY;
  }
  cv::Mat landmarks;
  landmarks_det.Compute(target, &landmarks, cv::Size(2, 5));
  std::vector<cv::Mat> images = {target};
  std::vector<cv::Mat> landmarks_vec = {landmarks};
  std::vector<cv::Rect> face_rects = {rect};
  //AlignFaces(&images, &landmarks_vec);
  AlignFaces2(&images, &landmarks_vec, image, face_rects);
  image_reid.Compute(images[0], &embedding);
  return RegistrationStatus::SUCCESS;
}

EmbeddingsGallery::EmbeddingsGallery(const std::string& ids_list,
                                     double threshold, DistaceAlgorithm algorithm, int min_size_fr,
                                     bool crop_gallery, const DetectorConfig &detector_config,
                                     const VectorCNN& landmarks_det,
                                     const VectorCNN& image_reid,
                                     bool use_greedy_matcher)
  : reid_threshold(threshold), reid_algorithm(algorithm),
    use_greedy_matcher(use_greedy_matcher) {
  if (ids_list.empty()) {
    return;
  }

  FaceDetection detector(detector_config);

  cv::FileStorage fs(ids_list, cv::FileStorage::Mode::READ);
  cv::FileNode fn = fs.root();
  int id = 0;
  for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit) {
    cv::FileNode item = *fit;
    std::string label = item.name();
    std::vector<cv::Mat> embeddings;

    // Please, note that the case when there are more than one image in gallery
    // for a person might not work properly with the current implementation
    // of the demo.
    // Remove this assert by your own risk.
    CV_Assert(item.size() == 1);

    for (size_t i = 0; i < item.size(); i++) {
      std::string path;
      if (file_exists(item[i].string())) {
        path = item[i].string();
      } else {
        path = folder_name(ids_list) + separator() + item[i].string();
      }

      cv::Mat image = cv::imread(path);
      CV_Assert(!image.empty());
      cv::Mat emb;
      RegistrationStatus status = RegisterIdentity(label, image, min_size_fr, crop_gallery,  detector, landmarks_det, image_reid, emb);
      if (status == RegistrationStatus::SUCCESS) {
        embeddings.push_back(emb);
        idx_to_id.push_back(id);
        std::cout << "label = " << label << " id = " << id << std::endl;
        identities.emplace_back(embeddings, label, id);
        ++id;
      }
    }
  }
}

std::vector<int> EmbeddingsGallery::GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const {
  if (embeddings.empty() || idx_to_id.empty())
    return std::vector<int>(embeddings.size(), unknown_id);

  cv::Mat distances(static_cast<int>(embeddings.size()), static_cast<int>(idx_to_id.size()), CV_32F);

  for (int i = 0; i < distances.rows; i++) {
    int k = 0;
    for (size_t j = 0; j < identities.size(); j++) {
      for (const auto& reference_emb : identities[j].embeddings) {
        distances.at<float>(i, k) = ComputeReidDistance(embeddings[i], reference_emb, reid_algorithm);
        k++;
      }
    }
  }
  KuhnMunkres matcher(use_greedy_matcher);
  auto matched_idx = matcher.Solve(distances);
  std::vector<int> output_ids;
  for (auto col_idx : matched_idx) {
    if (col_idx == 0xffffffff) {
      output_ids.push_back(unknown_id);
      continue;
    }
    float reid = distances.at<float>(output_ids.size(), col_idx);

    if (reid > reid_threshold) {
      output_ids.push_back(unknown_id);
      if (reid < (reid_threshold + 0.05)) {
        //std::cout << "####reid = " << reid << " : " << reid_threshold << " id = " << idx_to_id[col_idx] << " lable= " << GetLabelByID(idx_to_id[col_idx]) << std::endl;
      }
    } else {
      output_ids.push_back(idx_to_id[col_idx]);
      //std::cout << "****reid = " << reid << " : " << reid_threshold << " id = " << idx_to_id[col_idx] << " lable= " << GetLabelByID(idx_to_id[col_idx]) << std::endl;
    }
  }
  return output_ids;
}

std::string EmbeddingsGallery::GetLabelByID(int id) const {
  if (id >= 0 && id < static_cast<int>(identities.size()))
    return identities[id].label;
  else
    return unknown_label;
}
int EmbeddingsGallery::GetIDByID(int id) const {
  if (id >= 0 && id < static_cast<int>(identities.size()))
    return identities[id].id;
  else
    return -1;
}

size_t EmbeddingsGallery::size() const {
  return identities.size();
}

std::vector<std::string> EmbeddingsGallery::GetIDToLabelMap() const  {
  std::vector<std::string> map;
  map.reserve(identities.size());
  for (const auto& item : identities)  {
    map.emplace_back(item.label);
  }
  return map;
}

bool EmbeddingsGallery::LabelExists(const std::string& label) const {
  return identities.end() != std::find_if(identities.begin(), identities.end(),
  [label](const GalleryObject& o) {
    return o.label == label;
  });
}

void EmbeddingsGallery::updateIdentityDB(const std::vector<CIdentityParams> &params) {
  int id = identities.size();

  for (CIdentityParams param:params) {
    std::vector<cv::Mat> embeddings;
    for (auto &embedding:param.embeddings) {
      if (embedding.empty()) {
        continue;
      }
      cv::Mat emb(static_cast<int>(embedding.size()), 1, CV_32F);
      for (unsigned int i = 0; i < embedding.size(); i++) {
        emb.at<float>(i) = embedding[i];
      }

      embeddings.push_back(emb);
    }

    if (!embeddings.empty()) {
      idx_to_id.push_back(id);
      identities.emplace_back(embeddings, param.label, param.identityId);
      ++id;
    }
  }
}

