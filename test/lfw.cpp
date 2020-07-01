#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>

#include "detector.hpp"
#include "recognizer.hpp"
#include "ovface.h"

using namespace InferenceEngine;
using namespace ovface;
using namespace cv;
using namespace std::chrono;
using namespace std;

static const char help_message[] = "Print a usage message.";
static const char video_message[] = "Required. Path to a video or image file. Default value is \"cam\" to work with camera.";
static const char face_detection_model_message[] = "Required. Path to the Face Detection Retail model (.xml) file.";
static const char facial_landmarks_model_message[] = "Required. Path to the Facial Landmarks Regression Retail model (.xml) file.";
static const char face_reid_model_message[] = "Required. Path to the Face Reidentification Retail model (.xml) file.";
static const char target_device_message_face_reid[] = "Optional. Specify the target device for Face Reidentification Retail "
    "(the list of available devices is shown below).Default value is CPU. "
    "Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
    "The application looks for a suitable plugin for the specified device.";
static const char face_threshold_output_message[] = "Optional. Probability threshold for face detections.";
static const char threshold_output_message_face_reid[] = "Optional. Cosine distance threshold between two vectors for face reidentification.";
static const char reid_gallery_path_message[] = "Optional. Path to a faces gallery in .json format.";
static const char output_video_message[] = "Optional. File to write output video with visualization to.";
static const char min_size_fr_reg_output_message[] = "Optional. Minimum input size for faces during database registration.";
static const char detect_interval_output_message[] = "Optional. Detect Interval.";
static const char reid_interval_output_message[] = "Optional. Reidentification Interval.";
static const char input_image_height_output_message[] = "Optional. Input image height for face detector.";
static const char input_image_width_output_message[] = "Optional. Input image width for face detector.";

static const char face_detection_interval_message[] = "Optional. Face dection frame interval.";
static const char face_reid_interval_message[] = "Optional. Face reidentification frame interval.";

void showUsage();
bool initFaceDecAndRec();
void deInitFaceDecAndRec();
std::istream& safeGetline(std::istream& is, std::string& t);
std::vector<std::string> splitOneOf(const std::string& str, const std::string& delims, const size_t maxSplits);
double ComputeReidDistance(const cv::Mat& descr1, const cv::Mat& descr2, DistaceAlgorithm algorithm);

std::unique_ptr<AsyncDetection<DetectedObject>> s_fd;
std::unique_ptr<FaceRecognizerLfw> s_fr;
CVAChanParams s_params;

bool testPair2(const string & first, const string & second, double &cosdist, double &eucdist)
{
  cv::Mat frame1 = cv::imread(first, cv::IMREAD_COLOR);
  cv::Mat frame2 = cv::imread(second, cv::IMREAD_COLOR);

  if (frame1.channels() != 3) {
    cout << "picture " << first << " channels:" << frame1.channels() << endl;
    cout << "dims: " << frame1.dims << " depth: " << frame1.depth() << endl;
    return false;
  }
  if (frame2.channels() != 3) {
    cout << "picture " << second << " channels:" << frame2.channels() << endl;
    cout << "dims: " << frame2.dims << " depth: " << frame2.depth() << endl;
    return false;
  }

  std::vector<DetectedObject> faces1;
  std::vector<DetectedObject> faces2;
  std::vector<cv::Mat> embedings1;
  std::vector<cv::Mat> embedings2;

  // First picture
  s_fd->enqueue(frame1);
  if (frame1.channels() != 3) {
    cout << "picture " << first << " channels:" << frame1.channels()<<endl;
    return false;
  }
  s_fd->submitRequest();
  s_fd->wait();
  faces1 = s_fd->fetchResults();
  if (faces1.size() == 0) {
    cout << "Dectect no face in picture " << first << endl;
    return false;
  }
  
  embedings1 = s_fr->Recognize(frame1, faces1);

  // Second picture
  s_fd->enqueue(frame2);
  s_fd->submitRequest();
  s_fd->wait();
  faces2 = s_fd->fetchResults();
  if (faces2.size() == 0) {
    cout << "Dectect no face in picture " << second << endl;
    return false;
  }
  
  embedings2 = s_fr->Recognize(frame2, faces2);

  cosdist = ComputeReidDistance(embedings1[0], embedings2[0], DISTANCE_COSINE);
  eucdist = ComputeReidDistance(embedings1[0], embedings2[0], DISTANCE_EUCLIDEAN);

  return true;
}

bool testPair(const string & first, const string & second, double recogThreshold, bool expect)
{
  cv::Mat frame1 = cv::imread(first, cv::IMREAD_COLOR);
  cv::Mat frame2 = cv::imread(second, cv::IMREAD_COLOR);

  if (frame1.channels() != 3) {
    cout << "picture " << first << " channels:" << frame1.channels() << endl;
    cout << "dims: " << frame1.dims << " depth: " << frame1.depth() << endl;
    return false;
  }
  if (frame2.channels() != 3) {
    cout << "picture " << second << " channels:" << frame2.channels() << endl;
    cout << "dims: " << frame2.dims << " depth: " << frame2.depth() << endl;
    return false;
  }

  std::vector<DetectedObject> faces1;
  std::vector<DetectedObject> faces2;
  std::vector<cv::Mat> embedings1;
  std::vector<cv::Mat> embedings2;

  // First picture
  s_fd->enqueue(frame1);
  if (frame1.channels() != 3) {
    cout << "picture " << first << " channels:" << frame1.channels()<<endl;
    return false;
  }
  s_fd->submitRequest();
  s_fd->wait();
  faces1 = s_fd->fetchResults();
  if (faces1.size() == 0) {
    cout << "Dectect no face in picture " << first << endl;
    return false;
  }
  
  embedings1 = s_fr->Recognize(frame1, faces1);

  // Second picture
  s_fd->enqueue(frame2);
  s_fd->submitRequest();
  s_fd->wait();
  faces2 = s_fd->fetchResults();
  if (faces2.size() == 0) {
    cout << "Dectect no face in picture " << second << endl;
    return false;
  }
  
  embedings2 = s_fr->Recognize(frame2, faces2);

  double distance = ComputeReidDistance(embedings1[0], embedings2[0], s_params.distAlgorithm);

  bool matched = false;
  if (distance < recogThreshold) {
    matched = true;
  }

  return matched == expect;
}

void testLFW2() 
{
  if (!initFaceDecAndRec())
    return;

  std::vector<double> tpcosdists;
  std::vector<double> tpeucdists;
  std::vector<double> tncosdists;
  std::vector<double> tneucdists;
  
  ifstream fPairs("../share/pairs.txt");
  if (!fPairs.is_open()) {
    cout << "[ERROR]Open pairs.txt fail!" << endl;
    return;
  }
  
  string line;
  int rows = 0;
  while (!safeGetline(fPairs, line).eof()) {
    rows++;
    if (rows > 1) { // Ignore first line
      std::vector<std::string> splits = splitOneOf(line, "\t", 10);
      if (splits.size() == 3) { // Same face.
        int num1 = stoi(splits[1]);
        string snum1;
        if (num1 >= 100) {
          snum1 = "_0" + to_string(num1);
        }
        else if (num1 >= 10) {
          snum1 = "_00" + to_string(num1);
        }
        else {
          snum1 = "_000" + to_string(num1);
        }

        int num2 = stoi(splits[2]);
        string snum2;
        if (num2 >= 100) {
          snum2 = "_0" + to_string(num2);
        }
        else if (num2 >= 10) {
          snum2 = "_00" + to_string(num2);
        }
        else {
          snum2 = "_000" + to_string(num2);
        }

        string f1 = "../share/lfw/" + splits[0] + "/" + splits[0] + snum1 + ".jpg";
        string f2 = "../share/lfw/" + splits[0] + "/" + splits[0] + snum2 + ".jpg";
        double cosdist = 0.0;
        double eucdist = 0.0;
        bool ret = testPair2(f1, f2, cosdist, eucdist);
        if (ret) {
          tpcosdists.push_back(cosdist);
          tpeucdists.push_back(eucdist);
        }
      }
      else if (splits.size() == 4) { // Diff face.
        int num1 = stoi(splits[1]);
        string snum1;
        if (num1 >= 100) {
          snum1 = "_0" + to_string(num1);
        }
        else if (num1 >= 10) {
          snum1 = "_00" + to_string(num1);
        }
        else {
          snum1 = "_000" + to_string(num1);
        }

        int num2 = stoi(splits[3]);
        string snum2;
        if (num2 >= 100) {
          snum2 = "_0" + to_string(num2);
        }
        else if (num2 >= 10) {
          snum2 = "_00" + to_string(num2);
        }
        else {
          snum2 = "_000" + to_string(num2);
        }

        string f1 = "../share/lfw/" + splits[0] + "/" + splits[0] + snum1 + ".jpg";
        string f2 = "../share/lfw/" + splits[2] + "/" + splits[2] + snum2 + ".jpg";
        double cosdist = 0.0;
        double eucdist = 0.0;
        bool ret = testPair2(f1, f2, cosdist, eucdist);
        if (ret) {
          tncosdists.push_back(cosdist);
          tneucdists.push_back(eucdist);
        }
      }
    }
  }
  
  fPairs.close();
    
  double threshold=0.1;
  double threshold_step = 0.01;
  double threshold_max = 1.0;
  
  cout << "DISTANCE_COSINE: " << endl;
  cout << "TP \t TN \t Total \t Precision \t Threshold " << endl;
  while(threshold < threshold_max){
    int tp = 0; // True Positive
    int tn = 0; // True Negative
    
    for (size_t i = 0; i < tpcosdists.size(); i++) {
      if (tpcosdists[i] < threshold) {
        tp++;
      }
    }
    
    for (size_t i = 0; i < tncosdists.size(); i++) {
      if (tncosdists[i] >= threshold) {
        tn++;
      }
    }

    double accuracy = double(tp + tn) / double(rows - 1);
    cout << tp << " \t " << tn << " \t " << rows - 1 << " \t " << accuracy << " \t " << threshold << endl;
    threshold += threshold_step;
  }
  
  threshold=0.1;
  threshold_max = 1.24;
  cout << "DISTANCE_EUCLIDEAN: " << endl;
  cout << "TP \t TN \t Total \t Precision \t Threshold " << endl;
  while(threshold < threshold_max){
    int tp = 0; // True Positive
    int tn = 0; // True Negative
    
    for (size_t i = 0; i < tpeucdists.size(); i++) {
      if (tpeucdists[i] < threshold) {
        tp++;
      }
    }
    
    for (size_t i = 0; i < tneucdists.size(); i++) {
      if (tneucdists[i] >= threshold) {
        tn++;
      }
    }

    double accuracy = double(tp + tn) / double(rows - 1);
    cout << tp << " \t " << tn << " \t " << rows - 1 << " \t " << accuracy << " \t " << threshold << endl;
    threshold += threshold_step;
  }

  deInitFaceDecAndRec();
}

void testLFW() 
{
  if (!initFaceDecAndRec())
    return;

  double threshold=0.1;
  double threshold_step = 0.1;
  double threshold_max = 1.5;
  cout << "TP \t TN \t Total \t Precision \t Threshold " << endl;
  while(threshold< threshold_max){

    string line;
    int rows = 0;
    int tp = 0; // True Positive
    int tn = 0; // True Negative

    ifstream fPairs("../share/pairs.txt");
    if (!fPairs.is_open()) {
      cout << "[ERROR]Open pairs.txt fail!" << endl;
      return;
    }
    while (!safeGetline(fPairs, line).eof()) {
      rows++;
      //cout << rows << endl;

      if (rows > 1) { // Ignore first line
        std::vector<std::string> splits = splitOneOf(line, "\t", 10);
        if (splits.size() == 3) { // Same face.
          int num1 = stoi(splits[1]);
          string snum1;
          if (num1 >= 100) {
            snum1 = "_0" + to_string(num1);
          }
          else if (num1 >= 10) {
            snum1 = "_00" + to_string(num1);
          }
          else {
            snum1 = "_000" + to_string(num1);
          }

          int num2 = stoi(splits[2]);
          string snum2;
          if (num2 >= 100) {
            snum2 = "_0" + to_string(num2);
          }
          else if (num2 >= 10) {
            snum2 = "_00" + to_string(num2);
          }
          else {
            snum2 = "_000" + to_string(num2);
          }

          string f1 = "../share/lfw/" + splits[0] + "/" + splits[0] + snum1 + ".jpg";
          string f2 = "../share/lfw/" + splits[0] + "/" + splits[0] + snum2 + ".jpg";

          bool ret = testPair(f1, f2, threshold, true);
          if (ret) {
            tp++;
          }
        }
        else if (splits.size() == 4) { // Diff face.
          int num1 = stoi(splits[1]);
          string snum1;
          if (num1 >= 100) {
            snum1 = "_0" + to_string(num1);
          }
          else if (num1 >= 10) {
            snum1 = "_00" + to_string(num1);
          }
          else {
            snum1 = "_000" + to_string(num1);
          }

          int num2 = stoi(splits[3]);
          string snum2;
          if (num2 >= 100) {
            snum2 = "_0" + to_string(num2);
          }
          else if (num2 >= 10) {
            snum2 = "_00" + to_string(num2);
          }
          else {
            snum2 = "_000" + to_string(num2);
          }

          string f1 = "../share/lfw/" + splits[0] + "/" + splits[0] + snum1 + ".jpg";
          string f2 = "../share/lfw/" + splits[2] + "/" + splits[2] + snum2 + ".jpg";

          bool ret = testPair(f1, f2, threshold, false);
          if (ret) {
            tn++;
          }
        }
      }
      //threshold += threshold_step;
    }
    fPairs.close();

    double accuracy = double(tp + tn) / double(rows - 1);
    cout << tp << " \t " << tn << " \t " << rows - 1 << " \t " << accuracy << " \t " << threshold << endl;
    threshold += threshold_step;
  }

  deInitFaceDecAndRec();
}

int main(int argc, char* argv[]) {
  int i;  
  VAChannel::getDefVAChanParams(s_params);

  for (i = 1; i < argc; i++) {
    const char *pc = argv[i];
    if ((pc[0] == '-') && pc[1]) {
      if (!::strncmp(pc,"-psn_",5))
        continue;
      else if (!::strncmp(pc,"-dev", 4)) {
        s_params.device = argv[++i];
      } else if (!::strncmp(pc,"-m_fd", 5)) {
        s_params.faceDetectModelPath = argv[++i];
      } else if (!::strncmp(pc,"-m_lm", 5)) {
        s_params.landmarksModelPath = argv[++i];
      } else if (!::strncmp(pc,"-m_reid", 7)) {
        s_params.faceRecogModelPath = argv[++i];
      } else if (!::strncmp(pc,"-t_fd", 5)) {
        s_params.detectThreshold = atof(argv[++i]);
      } else if (!::strncmp(pc,"-t_reid", 7)) {
        s_params.reidThreshold = atof(argv[++i]);
      } else if (!::strncmp(pc,"-fg", 3)) {
        s_params.reidGalleryPath = argv[++i];
      } else if (!::strncmp(pc,"-min_size_fr", 12)) {
        s_params.minSizeHW = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-di", 3)) {
        s_params.detectInterval = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-ri", 3)) {
        s_params.reidInterval = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-inw_fd", 7)) {
        s_params.fdInImgWidth = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-inh_fd", 7)) {
        s_params.fdInImgHeight = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-h", 2)) {
        showUsage();
        return 0;
      }  else if (!::strncmp(pc,"-V", 2)) {
        std::cout << "OVFACE " << OVFACE_VERSION << std::endl;
        return 0;
      }
    }
  }

  try {
    testLFW2();
  } catch (const std::exception& error) {
    std::cout << error.what() << std::endl;
    return 1;
  } catch (...) {
    std::cout << "Unknown/internal exception happened." << std::endl;
    return 1;
  }

  std::cout << "Execution successful" << std::endl;

  return 0;
}




void showUsage() 
{
  std::cout << std::endl;
  std::cout << "ovface [OPTION]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << std::endl;
  std::cout << "    -h                             " << help_message << std::endl;
  std::cout << "    -i      '<path>'               " << video_message << std::endl;
  std::cout << "    -o      '<path>'               " << output_video_message << std::endl;
  std::cout << "    -m_fd   '<path>'               " << face_detection_model_message << std::endl;
  std::cout << "    -m_lm   '<path>'               " << facial_landmarks_model_message << std::endl;
  std::cout << "    -m_reid '<path>'               " << face_reid_model_message << std::endl;
  std::cout << "    -dev    '<device>'             " << target_device_message_face_reid << std::endl;
  std::cout << "    -t_fd                          " << face_threshold_output_message << std::endl;
  std::cout << "    -t_reid                        " << threshold_output_message_face_reid << std::endl;
  std::cout << "    -fg      '<path>'              " << reid_gallery_path_message << std::endl;
  std::cout << "    -min_size_fr                   " << min_size_fr_reg_output_message << std::endl;
  std::cout << "    -di                            " << detect_interval_output_message << std::endl;
  std::cout << "    -ri                            " << reid_interval_output_message << std::endl;
  std::cout << "    -inh_fd                        " << input_image_height_output_message << std::endl;
  std::cout << "    -inw_fd                        " << input_image_width_output_message << std::endl;
}

bool initFaceDecAndRec()
{
  const std::string fd_model_path = s_params.faceDetectModelPath;
  const std::string fr_model_path = s_params.faceRecogModelPath;
  const std::string lm_model_path = s_params.landmarksModelPath;
  std::string dismethod = (s_params.distAlgorithm == DISTANCE_EUCLIDEAN) ? "Euclidean" : "Cosine";

  std::cout << "Face detect model: " << fd_model_path<< " Threshold: " << s_params.detectThreshold<<std::endl;
  std::cout << "Face recognition model: " << fr_model_path << " Distance method: " << dismethod << std::endl;
  std::cout << "Face landmark model: " << lm_model_path << std::endl;

  std::string device = s_params.device;
  if (device == "")
    device = "CPU";

  std::cout << "Loading Inference Engine" << std::endl;
  Core ie;

  std::cout << "Device info: " << device << std::endl;
  std::cout << ie.GetVersions(device) << std::endl;
  if (device.find("CPU") != std::string::npos) {
    ie.SetConfig({ {PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES} }, "CPU");
  }
  else if (device.find("GPU") != std::string::npos) {
    ie.SetConfig({ {PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES} }, "GPU");
  }

  if (!fd_model_path.empty()) {
    // Load face detector
    DetectorConfig face_config(fd_model_path);
    face_config.deviceName = device;
    face_config.ie = ie;
    face_config.is_async = false;
    face_config.confidence_threshold = s_params.detectThreshold;
    face_config.networkCfg = s_params.networkCfg;
    face_config.increase_scale_x = 1.0;
    face_config.increase_scale_y = 1.0;
    s_fd.reset(new FaceDetection(face_config));
  }
  else {
    std::cout << "[ERROR]Face detect models are disabled!" << std::endl;
    return false;
  }

  if (!fr_model_path.empty() && !lm_model_path.empty()) {
    // Create face recognizer
    CnnConfig reid_config(fr_model_path);
    reid_config.deviceName = device;
    reid_config.networkCfg = s_params.networkCfg;
    if (checkDynamicBatchSupport(ie, device))
      reid_config.max_batch_size = s_params.maxBatchSize;
    else
      reid_config.max_batch_size = 1;
    reid_config.ie = ie;

    CnnConfig landmarks_config(lm_model_path);
    landmarks_config.deviceName = device;
    landmarks_config.networkCfg = s_params.networkCfg;
    if (checkDynamicBatchSupport(ie, device))
      landmarks_config.max_batch_size = s_params.maxBatchSize;
    else
      landmarks_config.max_batch_size = 1;
    landmarks_config.ie = ie;

    s_fr.reset(new FaceRecognizerLfw(landmarks_config, reid_config));
  }
  else {
    std::cout << "[ERROR]Face recognition models are disabled!" << std::endl;
    return false;
  }
  return true;
}

void deInitFaceDecAndRec()
{
  //TODO
}

std::istream& safeGetline(std::istream& is, std::string& t) 
{
  t.clear();

  // The characters in the stream are read one-by-one using a std::streambuf.
  // That is faster than reading them one-by-one using the std::istream.
  // Code that uses streambuf this way must be guarded by a sentry object.
  // The sentry object performs various tasks,
  // such as thread synchronization and updating the stream state.

  std::istream::sentry se(is, true);
  std::streambuf* sb = is.rdbuf();

  for (;;) {
    int c = sb->sbumpc();
    switch (c) {
    case '\n':
      return is;
    case '\r':
      if (sb->sgetc() == '\n')
        sb->sbumpc();
      return is;
    case std::streambuf::traits_type::eof():
      // Also handle the case when the last line has no line ending
      if (t.empty())
        is.setstate(std::ios::eofbit);
      return is;
    default:
      t += (char)c;
    }
  }
}
std::vector<std::string> splitOneOf(const std::string& str,
  const std::string& delims,
  const size_t maxSplits) {
  std::string remaining(str);
  std::vector<std::string> result;
  size_t splits = 0, pos;

  while (((maxSplits == 0) || (splits < maxSplits)) &&
    ((pos = remaining.find_first_of(delims)) != std::string::npos)) {
    result.push_back(remaining.substr(0, pos));
    remaining = remaining.substr(pos + 1);
    splits++;
  }

  if (remaining.length() > 0)
    result.push_back(remaining);

  return result;
}

double ComputeReidDistance(const cv::Mat& descr1, const cv::Mat& descr2, DistaceAlgorithm algorithm) {
  if (algorithm == DISTANCE_EUCLIDEAN) {
    double dist = 0.0f;
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
    dev_src(0) = std::max(static_cast<double>(std::numeric_limits<double>::epsilon()), dev_src(0));
    feature1 /= dev_src(0);
    cv::normalize(feature1, descr1);

    cv::Mat feature2(descr2);
    cv::meanStdDev(descr2, mean, dev_dst);
    dev_dst(0) = std::max(static_cast<double>(std::numeric_limits<double>::epsilon()), dev_dst(0));
    feature2 /= dev_dst(0);
    cv::normalize(feature2, descr2);
    cv::Mat diff = descr2 - descr1;
    dist = sqrt(diff.dot(diff));
    return dist;
  }
  else {
    double xy = static_cast<double>(descr1.dot(descr2));
    double xx = static_cast<double>(descr1.dot(descr1));
    double yy = static_cast<double>(descr2.dot(descr2));
    double norm = sqrt(xx) * sqrt(yy) + 1e-6f;
    return 1.0f - xy / norm;
  }
}
