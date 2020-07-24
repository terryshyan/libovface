// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "cnn.hpp"
#include "detector.hpp"


using namespace ovface;

enum class RegistrationStatus {
  SUCCESS,
  FAILURE_LOW_QUALITY,
  FAILURE_NOT_DETECTED,
};

struct GalleryObject {
    std::vector<cv::Mat> embeddings;
    std::string label;
    int id;

    GalleryObject(const std::vector<cv::Mat>& embeddings,
                  const std::string& label, int id)
        : embeddings(embeddings), label(label), id(id) {}
};

class EmbeddingsGallery {
public:
    static const char unknown_label[];
    static const int unknown_id;
    EmbeddingsGallery(const std::string& ids_list, double threshold, DistaceAlgorithm reid_algorithm, int min_size_fr,
                      bool crop_gallery, const DetectorConfig &detector_config,
                      const VectorCNN& landmarks_det,
                      const VectorCNN& image_reid,
                      bool use_greedy_matcher=false);
    size_t size() const;
    std::vector<int> GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const;
    std::string GetLabelByID(int id) const;
    int GetIDByID(int id) const;
    std::vector<std::string> GetIDToLabelMap() const;
    bool LabelExists(const std::string& label) const;
    void updateIdentityDB(const std::vector<CIdentityParams> &params);

private:
    RegistrationStatus RegisterIdentity(const std::string& identity_label,
                                        const cv::Mat& image,
                                        int min_size_fr,
                                        bool crop_gallery,
                                        FaceDetection& detector,
                                        const VectorCNN& landmarks_det,
                                        const VectorCNN& image_reid,
                                        cv::Mat & embedding);
    std::vector<int> idx_to_id;
    double reid_threshold;
    DistaceAlgorithm reid_algorithm;
    std::vector<GalleryObject> identities;
    bool use_greedy_matcher;
};

void RectangletoSquare(cv::Rect& rect, cv::Mat frame);
void RectangleAddMargin(cv::Rect& rect, cv::Mat frame, int mgn);
void AlignFaces(std::vector<cv::Mat>* face_images,
                std::vector<cv::Mat>* landmarks_vec);
void AlignFaces2(std::vector<cv::Mat>* face_images,
                std::vector<cv::Mat>* landmarks_vec, 
                const cv::Mat& frame, 
                std::vector<cv::Rect>& face_rects);

bool checkDynamicBatchSupport(const InferenceEngine::Core& ie, const std::string& device);
