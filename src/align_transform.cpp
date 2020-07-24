// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "face_reid.hpp"

#include <algorithm>
#include <vector>
#include <limits>
#include <opencv2/imgproc.hpp>

static const float h = 112.;
static const float w = 112.;
// reference landmarks points in the unit square [0,1]x[0,1]
static const float ref_landmarks_normalized[] = {
  38.2946f / w, 51.6963f / h, 73.5318f / w, 51.5014f / h, 56.0252f / w,
  71.7366f / h, 41.5493f / w, 92.3655f / h, 70.7299f / w, 92.2041f / h
};
  
void RectangletoSquare(cv::Rect& rect, cv::Mat frame) {
  int width = rect.width;
  int height = rect.height;
  if (width > height) {
    int offset = (width - height) / 2;
    if (rect.y - offset >= 0) {
      rect.y -= offset;
    } else {
      rect.y = 0;
    }

    if (rect.y + width >= frame.rows) {
      int offset2 = rect.y + width - frame.rows;
      if (rect.y - offset2 >= 0) {
        rect.y -= offset2;
      } else {
        rect.y = 0;
      }
      rect.height = frame.rows - rect.y;
    } else {
      rect.height = rect.width;
    }
  }
  else if (height > width) {
    int offset = (height - width) / 2;
    if (rect.x - offset >= 0) {
      rect.x -= offset;
    } else {
      rect.x = 0;
    }

    if (rect.x + height >= frame.cols) {
      int offset2 = rect.x + height - frame.cols;
      if (rect.x - offset2 >= 0) {
        rect.x -= offset2;
      } else {
        rect.x = 0;
      }
      rect.width = frame.cols - rect.x;
    } else {
      rect.width = rect.height;
    }
  }

  if (rect.height > rect.width) {
    rect.y += (rect.height - rect.width) / 2;
    rect.height = rect.width;
  }
}

void RectangleAddMargin(cv::Rect & rect, cv::Mat frame, int mgn) {
  int margin[4] = {mgn, mgn, mgn, mgn};
  int ox = rect.x;
  int oy = rect.y;
  if (rect.x - mgn > 0)
    rect.x = rect.x - mgn;
  else {
    margin[0] = rect.x;
    rect.x = 0;
  }
  
  if (rect.y - mgn > 0)
    rect.y = rect.y - mgn;
  else {
    margin[1] = rect.y;
    rect.y = 0;
  }
  
  if (ox + rect.width + mgn < frame.cols)
    rect.width += (mgn + margin[0]);
  else {
    margin[2] = frame.cols - rect.width - ox;
    rect.width += (margin[2] + margin[0]);
  }
  
  if (oy + rect.height + mgn < frame.rows)
    rect.height += (mgn + margin[1]);
  else {
    margin[3] = frame.rows - rect.height - oy;
    rect.height += (margin[3] + margin[1]);
  }
}

cv::Mat GetTransform(cv::Mat* src, cv::Mat* dst) {
  cv::Mat col_mean_src;
  reduce(*src, col_mean_src, 0, cv::REDUCE_AVG);
  for (int i = 0; i < src->rows; i++) {
    src->row(i) -= col_mean_src;
  }

  cv::Mat col_mean_dst;
  reduce(*dst, col_mean_dst, 0, cv::REDUCE_AVG);
  for (int i = 0; i < dst->rows; i++) {
    dst->row(i) -= col_mean_dst;
  }

  cv::Scalar mean, dev_src, dev_dst;
  cv::meanStdDev(*src, mean, dev_src);
  dev_src(0) =
    std::max(static_cast<double>(std::numeric_limits<float>::epsilon()), dev_src(0));
  *src /= dev_src(0);
  cv::meanStdDev(*dst, mean, dev_dst);
  dev_dst(0) =
    std::max(static_cast<double>(std::numeric_limits<float>::epsilon()), dev_dst(0));
  *dst /= dev_dst(0);

  cv::Mat w, u, vt;
  cv::SVD::compute((*src).t() * (*dst), w, u, vt);
  cv::Mat r = (u * vt).t();
  cv::Mat m(2, 3, CV_32F);
  m.colRange(0, 2) = r * (dev_dst(0) / dev_src(0));
  m.col(2) = (col_mean_dst.t() - m.colRange(0, 2) * col_mean_src.t());
  return m;
}

void AlignFaces(std::vector<cv::Mat>* face_images,
                std::vector<cv::Mat>* landmarks_vec) {
  if (landmarks_vec->size() == 0) {
    return;
  }
  CV_Assert(face_images->size() == landmarks_vec->size());
  cv::Mat ref_landmarks = cv::Mat(5, 2, CV_32F);

  for (size_t j = 0; j < face_images->size(); j++) {
    for (int i = 0; i < ref_landmarks.rows; i++) {
      ref_landmarks.at<float>(i, 0) =
        ref_landmarks_normalized[2 * i] * face_images->at(j).cols;
      ref_landmarks.at<float>(i, 1) =
        ref_landmarks_normalized[2 * i + 1] * face_images->at(j).rows;
      landmarks_vec->at(j).at<float>(i, 0) *= face_images->at(j).cols;
      landmarks_vec->at(j).at<float>(i, 1) *= face_images->at(j).rows;
    }
    cv::Mat m = GetTransform(&ref_landmarks, &landmarks_vec->at(j));
    cv::warpAffine(face_images->at(j), face_images->at(j), m, face_images->at(j).size(), cv::WARP_INVERSE_MAP);
  }
}

void AlignFaces2(std::vector<cv::Mat>* face_images, std::vector<cv::Mat>* landmarks_vec, 
                 const cv::Mat& frame, std::vector<cv::Rect>& face_rects) {
  if (landmarks_vec->size() == 0) {
    return;
  }
  CV_Assert(face_images->size() == landmarks_vec->size());
  cv::Mat ref_landmarks = cv::Mat(5, 2, CV_32F);

  for (size_t j = 0; j < face_images->size(); j++) {
    cv::Rect rect = face_rects.at(j);
    RectangleAddMargin(rect, frame, 16);
    RectangletoSquare(rect, frame);
    cv::Mat face_image(frame(rect));
    for (int i = 0; i < ref_landmarks.rows; i++) {
      ref_landmarks.at<float>(i, 0) =
        ref_landmarks_normalized[2 * i] * face_image.cols;
      ref_landmarks.at<float>(i, 1) =
        ref_landmarks_normalized[2 * i + 1] * face_image.rows;
      landmarks_vec->at(j).at<float>(i, 0) *= face_images->at(j).cols;
      landmarks_vec->at(j).at<float>(i, 1) *= face_images->at(j).rows;
      landmarks_vec->at(j).at<float>(i, 0) += face_rects.at(j).x - rect.x;
      landmarks_vec->at(j).at<float>(i, 1) += face_rects.at(j).y - rect.y;
    }
    cv::Mat m = GetTransform(&ref_landmarks, &landmarks_vec->at(j));
    cv::warpAffine(face_image, face_images->at(j), m, face_image.size(), cv::WARP_INVERSE_MAP);
  }
}

