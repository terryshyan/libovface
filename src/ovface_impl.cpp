#include <string>
#include <vector>
#include <ie_iextension.h>
#include "cnn.hpp"
#include "face_reid.hpp"
#include "image_grabber.hpp"
#include "ovface_impl.h"

using namespace InferenceEngine;
using namespace ovface;

#define OVMIN(a,b) ((a) > (b) ? (b) : (a))

static inline const float av_clipf(float a, float amin, float amax)
{
    if      (a < amin) return amin;
    else if (a > amax) return amax;
    else               return a;
}

static inline const float av_clip(int a, int amin, int amax)
{
    if      (a < amin) return amin;
    else if (a > amax) return amax;
    else               return a;
}

static double getSceneScore(cv::Mat prev_frame, cv::Mat frame, double &prev_mafd) {
  double ret = 0.0f;
  int w0 = prev_frame.cols;
  int h0 = prev_frame.rows;
  int w1 = frame.cols;
  int h1 = frame.rows;
  if (w0 == w1 && h0 == h1) {
    float mafd, diff;
    uint64 sad = 0;
    int nb_sad = 0;
    cv::Mat gray0(h0, w0, CV_8UC1);
    cv::Mat gray1(h0, w0, CV_8UC1);
    cv::cvtColor(prev_frame, gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame, gray1, cv::COLOR_BGR2GRAY);
    for (int i = 0; i < h0; i++) {
      for (int j = 0; j < w0; j++) {
        sad += abs(gray1.at<unsigned char>(i, j) - gray0.at<unsigned char>(i, j));
        nb_sad++;
      }
    }
    mafd = nb_sad ? (float)sad / nb_sad : 0;
    diff = fabs(mafd - prev_mafd);
    ret  = av_clipf(OVMIN(mafd, diff) / 100., 0, 1);
    prev_mafd = mafd;
  }

  return ret;
}

/*
static int getFrameDiff(cv::Mat frame0, cv::Mat frame1, int threshold) {
  int w0 = frame0.cols;
  int h0 = frame0.rows;
  int w1 = frame1.cols;
  int h1 = frame1.rows;
  int score = -1;
  if (w0 == w1 && h0 == h1) {
    cv::Mat gray0(h0, w0, CV_8UC1);
    cv::Mat gray1(h0, w0, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(h0, w0, CV_8UC1);
    cv::cvtColor(frame0, gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    for (int i = 0; i < h0; i++) {
      for (int j = 0; j < w0; j++) {
        int diff = abs(gray1.at<unsigned char>(i, j) - gray0.at<unsigned char>(i, j));
        if (diff > threshold) {
          mask.at<unsigned char>(i, j) = 1;
        } else {
          mask.at<unsigned char>(i, j) = 0;
        }
      }
    }
    score = cv::countNonZero(mask);
  }

  return score;
}
*/

bool checkDynamicBatchSupport(const Core& ie, const std::string& device)  {
  try  {
    if (ie.GetConfig(device, CONFIG_KEY(DYN_BATCH_ENABLED)).as<std::string>() != PluginConfigParams::YES)
      return false;
  } catch (const std::exception&)  {
    return false;
  }
  return true;
}

VAChannel *VAChannel::create(const CVAChanParams &params) {
  VAChannelImpl * t = new VAChannelImpl();
  std::cout << "VAChannel::create " << t << std::endl;
  if (t) {
    t->init(params);
    return t;
  }

  return nullptr;
}

int VAChannel::getDefVAChanParams(CVAChanParams &params) {
  params.device = "CPU";
  params.faceDetectModelPath = "./models/face-detection-adas-0001.xml";
  params.landmarksModelPath = "./models/landmarks-regression-retail-0009.xml";
  params.faceRecogModelPath = "./models/model-y1-0000.xml";
  params.reidGalleryPath = "./share/faces_gallery.json";
  params.networkCfg.nCpuThreadsNum = 0;
  params.networkCfg.bCpuBindThread = true;
  params.networkCfg.nCpuThroughputStreams = 1;
  params.detectThreshold = 0.7;
  params.reidThreshold = 0.55;
  params.trackerThreshold = 0.85;
  params.motionThreshold = 0.20;
  params.maxBatchSize = 16;
  params.minFaceArea = 900;
  params.distAlgorithm = DISTANCE_COSINE;
  params.detectMode = DETECT_MODE_VIDEO;
  params.forgetDelay = 150; //frame count
  params.showDelay = 5; //frequent count
  params.detectInterval = 1;
  params.reidInterval = 1;
  params.minSizeHW = 112;
  params.fdInImgWidth = 600;
  params.fdInImgHeight = 600;

  return 0;
}

void VAChannel::destroyed(VAChannel *pChan) {
  std::cout << "VAChannel::destroy " << pChan << std::endl;
  if (pChan) {
    VAChannel *tmp = pChan;
    pChan = nullptr;
    delete tmp;
  }
}

VAChannelImpl::VAChannelImpl()
  : m_prevMafd(0.0f)
  , m_frameid(0) {

}

VAChannelImpl::~VAChannelImpl() {
  std::cout << "~VAChannelImpl" << std::endl;
}

int VAChannelImpl::init(const CVAChanParams &param) {
  m_vaChanParams = param;
  const std::string fd_model_path = param.faceDetectModelPath;
  const std::string fr_model_path = param.faceRecogModelPath;
  const std::string lm_model_path = param.landmarksModelPath;

  std::string device = param.device;
  if (device == "")
    device = "CPU";

  std::cout << "Loading Inference Engine" << std::endl;
  Core ie;
  std::set<std::string> loadedDevices;
  std::cout << "Device info: " << device << std::endl;
  std::cout << ie.GetVersions(device) << std::endl;
  if (device.find("CPU") != std::string::npos) {
    ie.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}}, "CPU");
  } else if (device.find("GPU") != std::string::npos) {
    ie.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}}, "GPU");
  }

  loadedDevices.insert(device);

  if (!fd_model_path.empty()) {
    // Load face detector
    DetectorConfig face_config(fd_model_path);
    face_config.deviceName = device;
    face_config.ie = ie;
    face_config.is_async = true;
    face_config.confidence_threshold = param.detectThreshold;
    face_config.networkCfg = param.networkCfg;
    face_config.input_h = param.fdInImgHeight;
    face_config.input_w = param.fdInImgWidth;
    face_config.increase_scale_x = 1.15;
    face_config.increase_scale_y = 1.15;
    m_fd.reset(new FaceDetection(face_config));
  } else {
    m_fd.reset(new NullDetection<DetectedObject>);
  }

  if (!fd_model_path.empty() && !fr_model_path.empty() && !lm_model_path.empty()) {
    // Create face recognizer
    DetectorConfig face_registration_det_config(fd_model_path);
    face_registration_det_config.deviceName = device;
    face_registration_det_config.ie = ie;
    face_registration_det_config.is_async = false;
    face_registration_det_config.confidence_threshold = param.detectThreshold;
    face_registration_det_config.networkCfg = param.networkCfg;
    CnnConfig reid_config(fr_model_path);
    reid_config.deviceName = device;
    reid_config.networkCfg = param.networkCfg;
    if (checkDynamicBatchSupport(ie, device))
      reid_config.max_batch_size = param.maxBatchSize;
    else
      reid_config.max_batch_size = 1;
    reid_config.ie = ie;

    CnnConfig landmarks_config(lm_model_path);
    landmarks_config.deviceName = device;
    landmarks_config.networkCfg = param.networkCfg;
    if (checkDynamicBatchSupport(ie, device))
      landmarks_config.max_batch_size = param.maxBatchSize;
    else
      landmarks_config.max_batch_size = 1;
    landmarks_config.ie = ie;

    m_fr.reset(new FaceRecognizerDefault(
                 landmarks_config, reid_config,
                 face_registration_det_config,
                 param.reidGalleryPath, param.reidThreshold, param.distAlgorithm, param.minSizeHW, true, false));
  } else {
    std::cout << "Face recognition models are disabled!" << std::endl;
    m_fr.reset(new FaceRecognizerNull);
  }

  // Create tracker for reid
  TrackerParams tracker_reid_params;
  tracker_reid_params.min_track_duration = 1;
  tracker_reid_params.forget_delay = param.forgetDelay;
  tracker_reid_params.affinity_thr = param.trackerThreshold;
  tracker_reid_params.averaging_window_size_for_rects = 1;
  tracker_reid_params.averaging_window_size_for_labels = std::numeric_limits<int>::max();
  tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
  tracker_reid_params.drop_forgotten_tracks = true;
  tracker_reid_params.max_num_objects_in_track = std::numeric_limits<int>::max();
  tracker_reid_params.objects_type = "face";
  tracker_reid_params.max_frequent_count = param.showDelay;
  m_tracker.reset(new Tracker(tracker_reid_params));

  return 0;
}

int VAChannelImpl::setIdentityDB(const std::vector<CIdentityParams> &params) {
  if (m_fr && params.size() > 0) {
    m_fr->updateIdentityDB(params);
  }

  return 0;
}

int VAChannelImpl::process(const CFrameData &frameData, std::vector<CResult> &results, bool bForce) {
  int ret = -1;
  if (frameData.pFrame == NULL)
    return ret;
  
  DetectedObjects faces;
  std::vector<int> ids;
  if (m_vaChanParams.detectInterval == 0 || m_frameid % m_vaChanParams.detectInterval == 0 || bForce) {
    if (frameData.format == FRAME_FOMAT_I420) {
      cv::Mat yuv(frameData.height + frameData.height/2, frameData.width, CV_8UC1, frameData.pFrame);
      cv::Mat bgr(frameData.height, frameData.width, CV_8UC3);
      cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
      m_frame = bgr;
    } else if (frameData.format == FRAME_FOMAT_RGB) {
      cv::Mat rgb(frameData.height, frameData.width, CV_8UC3, frameData.pFrame);
      cv::Mat bgr(frameData.height, frameData.width, CV_8UC3);
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      m_frame = bgr;
    } else if (frameData.format == FRAME_FOMAT_BGR) {
      cv::Mat bgr(frameData.height, frameData.width, CV_8UC3, frameData.pFrame);
      m_frame = bgr;
    } else {
      return ret;
    }
    
    ret = 0;
    bool bNeedDetect = true;
    double totalscore = getSceneScore(m_prevframe, m_frame, m_prevMafd);
    double threshold = m_vaChanParams.motionThreshold / ((m_frame.cols * m_frame.rows) / (m_vaChanParams.minSizeHW * m_vaChanParams.minSizeHW));
    if (m_frameid > 0 && !bForce && totalscore < threshold) {
      bNeedDetect = false;
    }
    
    if (bNeedDetect) {
      cv::Mat prevframe;
      if (m_prevframe.rows > 0 || m_prevframe.cols > 0)
        prevframe = m_prevframe.clone();
      auto started = std::chrono::high_resolution_clock::now();
      m_prevframe = m_frame.clone();
      m_fd->enqueue(m_frame);
      m_fd->submitRequest();
      m_fd->wait();
      faces = m_fd->fetchResults();
      //std::cout << "***** face detect faces.size() = " << faces.size() << std::endl;
      if (m_vaChanParams.reidInterval == 0 ||
          (m_vaChanParams.reidInterval > 0 && (m_frameid % m_vaChanParams.reidInterval == 0))) {
        DetectedObjects noneedreidfaces;
		DetectedObjects needreidfaces;
        for (size_t i = 0; i < faces.size(); i++) {
          double mafd = 0.f;
          double score = 0.f;
          if (prevframe.rows > 0 || prevframe.cols > 0)
            score = getSceneScore(prevframe(faces[i].rect), m_frame(faces[i].rect), mafd);
          std::cout << i << "*********** score = " << score << std::endl;
          if (bForce || score >= m_vaChanParams.motionThreshold) {
            needreidfaces.push_back(faces[i]);
          } else {
            noneedreidfaces.push_back(faces[i]);
          }
        }
        ids = m_fr->Recognize(m_frame, needreidfaces);
        faces.clear();
        for (size_t i = 0; i < needreidfaces.size(); i++) {
          faces.push_back(needreidfaces[i]);
        }
        for (size_t i = 0; i < noneedreidfaces.size(); i++) {
          faces.push_back(noneedreidfaces[i]);
          ids.push_back(TrackedObject::UNKNOWN_LABEL_IDX);
        }
      } else {
        for (size_t i = 0; i < faces.size(); i++) {
          ids.push_back(TrackedObject::UNKNOWN_LABEL_IDX);
        }
      }
    } else {
      faces = m_lastObjects;
      ids = m_lastIds;
    }
  } else {
    faces = m_lastObjects;
    ids = m_lastIds;
  }

  TrackedObjects tracked_face_objects;
  for (size_t i = 0; i < faces.size(); i++) {
    tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, ids[i]);
    //std::cout << "***** face recognize " << i << " idx = " << ids[i] << std::endl;
  }

  m_tracker->Process(m_frame, tracked_face_objects, m_frameid);

  const TrackedObjects tracked_faces = m_tracker->TrackedDetectionsWithLabels();
  for (size_t j = 0; j < tracked_faces.size(); j++) {
    const TrackedObject& face = tracked_faces[j];
    std::string face_label = m_fr->GetLabelByID(face.label);
    int identityId = -1;
    std::string label_to_draw;
    if (face.label != EmbeddingsGallery::unknown_id) {
      label_to_draw += face_label;
      identityId = m_fr->GetIDByID(face.label);
    }

    CResult result;
    result.rect.left = face.rect.x;
    result.rect.top = face.rect.y;
    result.rect.right = face.rect.x + face.rect.width;
    result.rect.bottom = face.rect.y + face.rect.height;
    result.frameId = m_frameid;
    result.label = label_to_draw;
    result.identityId = identityId;
    result.trackId = face.object_id;
    results.push_back(result);
  }
  
  m_lastObjects = faces;
  m_lastIds = ids;
  m_frameid++;

  return ret;
}

int VAChannelImpl::fetchImageEmbedding(const CFrameData &frameData, std::vector<float> &embedding) {
  if (frameData.pFrame == NULL)
    return -1;
  
  cv::Mat frame;
  DetectedObjects faces;
  std::vector<int> ids;
  if (frameData.format == FRAME_FOMAT_I420) {
    cv::Mat yuv(frameData.height + frameData.height/2, frameData.width, CV_8UC1, frameData.pFrame);
    cv::Mat bgr(frameData.height, frameData.width, CV_8UC3);
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
    frame = bgr;
  } else if (frameData.format == FRAME_FOMAT_RGB) {
    cv::Mat rgb(frameData.height, frameData.width, CV_8UC3, frameData.pFrame);
    cv::Mat bgr(frameData.height, frameData.width, CV_8UC3);
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    frame = bgr;
  } else if (frameData.format == FRAME_FOMAT_BGR) {
    cv::Mat bgr(frameData.height, frameData.width, CV_8UC3, frameData.pFrame);
    frame = bgr;
  } else {
    return -1;
  }
  
  m_fd->enqueue(frame);
  m_fd->submitRequest();
  m_fd->wait();
  faces = m_fd->fetchResults();
  
  if (faces.size() == 0) {
    std::cout << "Dectect no face in this frame!" << std::endl;
    return -1;
  }
  
  std::vector<cv::Mat> embeddings;
  std::vector<DetectedObject> tempfaces;
  DetectedObject face1 = faces[0];
	for (size_t i = 1; i < faces.size(); i++) {
    DetectedObject face = faces[i];
		if (face.confidence >= m_vaChanParams.detectThreshold && face.rect.area() > face1.rect.area())
			face1 = face;
	}
  
  tempfaces.push_back(face1);
  embeddings = m_fr->Recognize2(frame, tempfaces);
  
  if (embeddings.size() <= 0)
    return -1;
  
  embedding = std::vector<float>(embeddings[0].reshape(1, 1));
  
  return 0;
}

int VAChannelImpl::fetchImageEmbedding(const unsigned char *imgData, int imgSize, std::vector<float> &embedding) {
  if (imgData == NULL || imgSize <= 0)
    return -1;
  
  cv::InputArray in {imgData,imgSize};
  cv::Mat frame=cv::imdecode(in, cv::IMREAD_COLOR);

  DetectedObjects faces;
  std::vector<int> ids;
  m_fd->enqueue(frame);
  m_fd->submitRequest();
  m_fd->wait();
  faces = m_fd->fetchResults();
  
  if (faces.size() == 0) {
    std::cout << "Dectect no face in this frame!" << std::endl;
    return -1;
  }
  
  std::vector<cv::Mat> embeddings;
  std::vector<DetectedObject> tempfaces;
  DetectedObject face1 = faces[0];
	for (size_t i = 1; i < faces.size(); i++) {
    DetectedObject face = faces[i];
		if (face.confidence >= m_vaChanParams.detectThreshold && face.rect.area() > face1.rect.area())
			face1 = face;
	}
  
  tempfaces.push_back(face1);
  embeddings = m_fr->Recognize2(frame, tempfaces);
  
  if (embeddings.size() <= 0)
    return -1;
  
  embedding = std::vector<float>(embeddings[0].reshape(1, 1));
  
  return 0;
}

int VAChannelImpl::fetchImageEmbedding(const char *filename, std::vector<float> &embedding) {
  cv::Mat frame = cv::imread(filename, cv::IMREAD_COLOR);
  if (frame.channels() != 3) {
    std::cout << "picture " << filename << " channels:" << frame.channels() << std::endl;
    std::cout << "dims: " << frame.dims << " depth: " << frame.depth() << std::endl;
    return -1;
  }

  std::vector<DetectedObject> faces;
  std::vector<cv::Mat> embeddings;

  m_fd->enqueue(frame);
  m_fd->submitRequest();
  m_fd->wait();
  faces = m_fd->fetchResults();
  if (faces.size() == 0) {
    std::cout << "Dectect no face in picture!" << filename << std::endl;
    return -1;
  }

  std::vector<DetectedObject> tempfaces;
  DetectedObject face1 = faces[0];
	for (size_t i = 1; i < faces.size(); i++) {
    DetectedObject face = faces[i];
		if (face.confidence >= m_vaChanParams.detectThreshold && face.rect.area() > face1.rect.area())
			face1 = face;
	}
  
  tempfaces.push_back(face1);
  embeddings = m_fr->Recognize2(frame, tempfaces);
  
  if (embeddings.size() <= 0)
    return -1;
  
  embedding = std::vector<float>(embeddings[0].reshape(1, 1));

  return 0;
}


