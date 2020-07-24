// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <algorithm>
#include <utility>
#include <ie_iextension.h>

#include "cnn.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "image_grabber.hpp"
#include "logger.hpp"


namespace ovface {

class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;

    virtual bool LabelExists(const std::string &label) const = 0;
    virtual std::string GetLabelByID(int id) const = 0;
    virtual int GetIDByID(int id) const = 0;
    virtual std::vector<std::string> GetIDToLabelMap() const = 0;

    virtual std::vector<int> Recognize(const cv::Mat& frame, const DetectedObjects& faces) = 0;
    virtual std::vector<cv::Mat> Recognize2(const cv::Mat& frame, const DetectedObjects& faces) = 0;
    
    virtual void PrintPerformanceCounts(
        const std::string &landmarks_device, const std::string &reid_device) = 0;
    virtual void updateIdentityDB(const std::vector<CIdentityParams> &params) = 0;
};

class FaceRecognizerNull : public FaceRecognizer {
public:
    bool LabelExists(const std::string &) const override { return false; }

    std::string GetLabelByID(int) const override {
        return EmbeddingsGallery::unknown_label;
    }

    int GetIDByID(int) const override {
        return -1;
    }
    
    std::vector<std::string> GetIDToLabelMap() const override { return {}; }

    std::vector<int> Recognize(const cv::Mat&, const DetectedObjects& faces) override {
        return std::vector<int>(faces.size(), EmbeddingsGallery::unknown_id);
    }

    std::vector<cv::Mat> Recognize2(const cv::Mat& frame, const DetectedObjects& faces) override{
        return std::vector<cv::Mat>(faces.size(), cv::Mat());
    }
    
    void PrintPerformanceCounts(
        const std::string &, const std::string &) override {}
    void updateIdentityDB(const std::vector<CIdentityParams> &params) override {}
};

class FaceRecognizerDefault : public FaceRecognizer {
public:
    FaceRecognizerDefault(
            const CnnConfig& landmarks_detector_config,
            const CnnConfig& reid_config,
            const DetectorConfig& face_registration_det_config,
            const std::string& face_gallery_path,
            double reid_threshold,
            DistaceAlgorithm reid_algorithm,
            int min_size_fr,
            bool crop_gallery,
            bool greedy_reid_matching
    );

    bool LabelExists(const std::string &label) const override;

    std::string GetLabelByID(int id) const override;

    int GetIDByID(int id) const override;
    
    std::vector<std::string> GetIDToLabelMap() const override;

    std::vector<int> Recognize(const cv::Mat& frame, const DetectedObjects& faces) override;
    
    std::vector<cv::Mat> Recognize2(const cv::Mat& frame, const DetectedObjects& faces) override;

    void PrintPerformanceCounts(const std::string &landmarks_device, const std::string &reid_device);

    void updateIdentityDB(const std::vector<CIdentityParams> &params);

private:
    VectorCNN landmarks_detector;
    VectorCNN face_reid;
    EmbeddingsGallery face_gallery;
};

// For LFW test
class FaceRecognizerLfw {
public:
  FaceRecognizerLfw(
    const CnnConfig& landmarks_detector_config,
    const CnnConfig& reid_config
  );
  // Return embeddings
  std::vector<cv::Mat> Recognize(const cv::Mat& frame, const DetectedObjects& faces);

private:
  VectorCNN landmarks_detector;
  VectorCNN face_reid;
};


};  // namespace ovface
