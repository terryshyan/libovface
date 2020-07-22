/*
 * ovface_impl.h
 *
 */

#ifndef __OVFACE_IMPL_H
#define __OVFACE_IMPL_H

#include "ovface.h"

#include "tracker.hpp"
#include "detector.hpp"
#include "recognizer.hpp"

namespace ovface {

class VAChannelImpl:public VAChannel {
public:
  VAChannelImpl();
  ~VAChannelImpl();

  int init(const CVAChanParams &param);
  virtual int setIdentityDB(const std::vector<CIdentityParams> &params);
  virtual int process(const CFrameData &frameData, std::vector<CResult> &results, bool bForce = false);

private:
  CVAChanParams m_vaChanParams;
  std::vector<CIdentityParams> m_identityParams;
  std::unique_ptr<AsyncDetection<DetectedObject>> m_fd;
  std::unique_ptr<FaceRecognizer> m_fr;
  std::unique_ptr<Tracker> m_tracker;
  cv::Mat m_frame;
  cv::Mat m_prevframe;
  double m_prevMafd;
  int m_frameid;
  DetectedObjects m_lastObjects;
  std::vector<int> m_lastIds;
};

};
#endif

