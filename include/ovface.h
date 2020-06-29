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

namespace ovface {

#define OVFACE_VERSION "0.0.1"

enum FrameFormat {
  FRAME_FOMAT_I420  = 0x00,
  FRAME_FOMAT_BGR   = 0x01,
  FRAME_FOMAT_RGB   = 0x02
};

enum DetectMode {
  DETECT_MODE_VIDEO = 0x00,
  DETECT_MODE_IMAGE = 0x01
};

enum DistaceAlgorithm {
  DISTANCE_EUCLIDEAN  = 0x00,
  DISTANCE_COSINE     = 0x01
};

struct CRect {
  int left;
  int top;
  int right;
  int bottom;
};

struct CResult {
  CRect rect;
  std::string label;
  int trackId;
  int frameId;
};

struct CIdentityParams {
  std::string label;
  std::string identityId;
  std::vector<std::vector<float>> embeddings;
};

struct CNetWorkCPUConfig {
  int nCpuThreadsNum;         //default 0
  bool bCpuBindThread;        //default true
  int nCpuThroughputStreams;  //default 1
};

struct CVAChanParams {
  std::string device;
  std::string faceDetectModelPath;
  std::string landmarksModelPath;
  std::string faceRecogModelPath;
  std::string reidGalleryPath;
  CNetWorkCPUConfig networkCfg;
  float detectThreshold;
  float reidThreshold;
  float trackerThreshold;
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

struct CFrameData {
  unsigned char *pFrame;
  int width;
  int height;
  FrameFormat format;
};

class VAChannel {
public:
  static VAChannel *create(const CVAChanParams &params);
  static int getDefVAChanParams(CVAChanParams &params);
  static void destroyed(VAChannel *pChan);
  virtual ~VAChannel() {};
  virtual int setIdentityDB(const std::vector<CIdentityParams> &params) = 0;
  virtual int process(const CFrameData &frameData, std::vector<CResult> &results, bool bForce = false) = 0;
};
};
#endif

