#include <string>
#include <vector>
#include <ie_iextension.h>
#include "cnn.hpp"
#include "face_reid.hpp"
#include "image_grabber.hpp"
#include "ovface_impl.h"

using namespace InferenceEngine;
using namespace ovface;

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
  params.networkCfg.nCpuThreadsNum = 4;
  params.networkCfg.bCpuBindThread = true;
  params.networkCfg.nCpuThroughputStreams = 1;
  params.detectThreshold = 0.7;
  params.reidThreshold = 0.55;
  params.trackerThreshold = 0.85;
  params.maxBatchSize = 16;
  params.minFaceArea = 900;
  params.distAlgorithm = DISTANCE_COSINE;
  params.detectMode = DETECT_MODE_VIDEO;
  params.forgetDelay = 150;
  params.showDelay = 30;
  params.detectInterval = 1;
  params.reidInterval = 1;
  params.minSizeHW = 112;
  params.cropGallery = false;
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
  : m_frameid(0) {

}

VAChannelImpl::~VAChannelImpl() {
  std::cout << "~VAChannelVino" << std::endl;
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
    //face_config.input_h = param.fdInImgHeight;
    //face_config.input_w = param.fdInImgWidth;
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
                 param.reidGalleryPath, param.reidThreshold, param.distAlgorithm, 
                 param.minSizeHW, param.cropGallery, false));
    
  } else {
    std::cout << "FaceRecognizerNull models are created." << std::endl;
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
    if (m_frameid > 0)
      m_prevframe = m_frame;

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

    m_fd->enqueue(m_frame);
    m_fd->submitRequest();
    m_fd->wait();
    
    faces = m_fd->fetchResults();
    if (m_vaChanParams.reidInterval == 0 ||
        (m_vaChanParams.reidInterval > 0 && (m_frameid % m_vaChanParams.reidInterval == 0))) {
      ids = m_fr->Recognize(m_frame, faces);
    } else {
      for (size_t i = 0; i < faces.size(); i++) {
        ids.push_back(TrackedObject::UNKNOWN_LABEL_IDX);
      }
    }
  } else {
    faces = m_lastObjects;
    ids = m_lastIds;
  }

  TrackedObjects tracked_face_objects;
  for (size_t i = 0; i < faces.size(); i++) {
    tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, ids[i]);
  }

  m_tracker->Process(m_frame, tracked_face_objects, m_frameid);

  const TrackedObjects tracked_faces = m_tracker->TrackedDetectionsWithLabels();
  for (size_t j = 0; j < tracked_faces.size(); j++) {
    const TrackedObject& face = tracked_faces[j];
    std::string face_label = m_fr->GetLabelByID(face.label);

    std::string label_to_draw;
    if (face.label != EmbeddingsGallery::unknown_id)
      label_to_draw += face_label;

    CResult result;
    result.rect.left = face.rect.x;
    result.rect.top = face.rect.y;
    result.rect.right = face.rect.x + face.rect.width;
    result.rect.bottom = face.rect.y + face.rect.height;
    result.frameId = m_frameid;
    result.label = label_to_draw;
    result.trackId = face.object_id;
    results.push_back(result);
  }
  
  m_lastObjects = faces;
  m_lastIds = ids;
  m_frameid++;

  return 0;
}


