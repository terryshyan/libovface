/*
 * ovface.h
 *
 */

#ifndef __OVFACE_H
#define __OVFACE_H

#include <string>
#include <memory>
#include <vector>
#include <utility>

#ifdef LIBOVFACE_EXPORTS
#define OVFACE_API __declspec(dllexport)
#else
#define OVFACE_API
#endif

namespace ovface {

#define OVFACE_VERSION "0.0.1"

enum OVFACE_API FrameFormat {
  FRAME_FOMAT_I420  = 0x00,
  FRAME_FOMAT_BGR   = 0x01,
  FRAME_FOMAT_RGB   = 0x02
};

enum OVFACE_API DetectMode {
  DETECT_MODE_VIDEO = 0x00,
  DETECT_MODE_IMAGE = 0x01
};

enum DistaceAlgorithm {
  DISTANCE_EUCLIDEAN  = 0x00,
  DISTANCE_COSINE     = 0x01
};

struct OVFACE_API CRect {
  int left;
  int top;
  int right;
  int bottom;
};

struct OVFACE_API CResult {
  CRect rect;
  std::string label;
  int identityId;
  int trackId;
  int frameId;
};

struct OVFACE_API CIdentityParams {
  std::string label;
  int identityId;
  std::vector<std::vector<float>> embeddings;
};

struct OVFACE_API CNetWorkCPUConfig {
  int nCpuThreadsNum;         //default 0
  bool bCpuBindThread;        //default true
  int nCpuThroughputStreams;  //default 1
};

struct OVFACE_API CVAChanParams {
  std::string device;
  std::string faceDetectModelPath;
  std::string landmarksModelPath;
  std::string faceRecogModelPath;
  std::string reidGalleryPath;
  CNetWorkCPUConfig networkCfg;
  float detectThreshold;
  float reidThreshold;
  float trackerThreshold;
  float motionThreshold;
  int maxBatchSize;
  int minFaceArea;
  DistaceAlgorithm distAlgorithm;
  DetectMode detectMode;
  int forgetDelay;
  int showDelay;
  int detectInterval;
  int reidInterval;
  int minSizeHW;
  int fdInImgWidth;
  int fdInImgHeight;
};

struct OVFACE_API CFrameData {
  unsigned char *pFrame;
  int width;
  int height;
  FrameFormat format;
};

class OVFACE_API VAChannel {
public:
  static VAChannel *create(const CVAChanParams &params);
  static int getDefVAChanParams(CVAChanParams &params);
  static void destroyed(VAChannel *pChan);
  virtual ~VAChannel() {};
  virtual int setIdentityDB(const std::vector<CIdentityParams> &params) = 0;
  virtual int process(const CFrameData &frameData, std::vector<CResult> &results, bool bForce = false) = 0;
  virtual int fetchImageEmbedding(const CFrameData &frameData, std::vector<float> &embedding) = 0;
  virtual int fetchImageEmbedding(const unsigned char *imgData, int imgSize, std::vector<float> &embedding) = 0;
  virtual int fetchImageEmbedding(const char *filename, std::vector<float> &embedding) = 0;
};
};
#endif

