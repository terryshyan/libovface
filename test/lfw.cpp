#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "ovface.h"


using namespace ovface;

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

cv::Mat getCropped(cv::Mat input, Rect rectangle, int crop_size, int style);
bool testPair(const string & first, const string & second, int recogThreshold, bool expect, bool * ignore);

static void showUsage() {
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

void testLFW() {
  VINOLibParam libParam;
  libParam.device = "CPU";
  libParam.faceDetectModelPath = "face-detection-adas-0001.xml";
  libParam.faceRecogModelPath = "face-reidentification-retail-0095.xml";
  libParam.landmarksModelPath = "landmarks-regression-retail-0009.xml";
  libParam.detectThreshold = 60;
  libParam.maxNetworks = 2;
  libParam.threadNum = 1;
  libParam.minFaceAreaDiff = 900;
  initVinoLib(libParam);

  vector<int> thresholds = { 0,10,20,30,40,50,60,70,80,90,100 };
  //vector<int> thresholds = {0};
  cout << "TP \t TN \t Total \t Precision \t Threshold " << endl;
  for (auto & thd : thresholds) {
    string line;
    int rows = 0;
    int ignoreCount = 0;
    int tp = 0;
    int tn = 0;

    ifstream fPairs("lfw/pairs.txt");
    if (!fPairs.is_open()) {
      cout << "Open lfw/pairs.txt fail!" << endl;
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

          string f1 = "lfw/" + splits[0] + "/" + splits[0] + snum1 + ".jpg";
          string f2 = "lfw/" + splits[0] + "/" + splits[0] + snum2 + ".jpg";

          bool ignore = false;
          bool ret = testPair(f1, f2, thd, true, &ignore);
          if (ignore) {
            ignoreCount++;
            continue;
          }
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

          string f1 = "lfw/" + splits[0] + "/" + splits[0] + snum1 + ".jpg";
          string f2 = "lfw/" + splits[2] + "/" + splits[2] + snum2 + ".jpg";

          bool ignore = false;
          bool ret = testPair(f1, f2, thd, false, &ignore);
          if (ignore) {
            ignoreCount++;
            continue;
          }
          if (ret) {
            tn++;
          }
        }
      }
    }
    fPairs.close();

    float precision = (tp + tn) / ((rows - 1 - ignoreCount)*1.0);
    cout << tp << " \t " << tn << " \t " << rows - 1 - ignoreCount << " \t " << precision << " \t " << thd << endl;
  }

  uninitVinoLib();
}


void testLib(CVAChanParams &param, const char *src, const char *dst) {
  float work_time_ms = 0.f;
  size_t work_num_frames = 0;

  //----Capture
  cv::VideoCapture capture;
  capture.open(src);
  if (!capture.isOpened()) {
      std::cout << "[ERROR] Fail to capture video." << std::endl;
      return;
  }
  int w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

  //----Create chan
  param.fdInImgWidth = w;
  param.fdInImgHeight = h;
  VAChannel *chan =VAChannel::create(param);

  cv::VideoWriter writer;
  writer.open(dst,cv::VideoWriter::fourcc('X','2','6','4'),30,cv::Size(w,h));

  while (true) {
    cv::Mat frame;
    capture >> frame;

    if (frame.empty()) {
      std::cout << " Finished" << std::endl;
      break;
    }
    cv::Mat frame2 = frame.clone();
    CFrameData frameData;
    frameData.format = FRAME_FOMAT_BGR;
    frameData.pFrame = frame.data;
    frameData.width = w;
    frameData.height = h;
    std::vector<CResult> results;
    auto started = std::chrono::high_resolution_clock::now();
    chan->process(frameData, results);
    auto elapsed = std::chrono::high_resolution_clock::now() - started;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    work_time_ms += elapsed_ms;
    ++work_num_frames;

    //std::cout << results.size() << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
      CResult result = results[i];
      cv::Rect rect;
      rect.x = result.rect.left;
      rect.y = result.rect.top;
      rect.height = result.rect.bottom - result.rect.top;
      rect.width = result.rect.right - result.rect.left;
      DrawObject(frame2, rect, result.label);
      //std::cout << result.frameId << "/" << i << " rect=" << result.rect.left << "," << result.rect.top << " " << result.rect.right << ":" << result.rect.bottom << "   " << result.label << std::endl;
    }
    writer.write(frame2);
  }

  float fps = 1e3f / (work_time_ms / static_cast<float>(work_num_frames) + 1e-6f);
  std::cout << std::to_string(static_cast<int>(fps)) << " FPS" << std::endl;

  VAChannel::destroyed(chan);
  writer.release();
}

int main(int argc, char* argv[]) {
  int i;
  CVAChanParams chanParams;
  VAChannel::getDefVAChanParams(chanParams);
  const char *src = "./share/test.mp4";
  const char *dst = "./share/out_v.mp4";

  for (i = 1; i < argc; i++) {
    const char *pc = argv[i];
    if ((pc[0] == '-') && pc[1]) {
      if (!::strncmp(pc,"-psn_",5))
        continue;
      else if (!::strncmp(pc,"-dev", 4)) {
        chanParams.device = argv[++i];
      } else if (!::strncmp(pc,"-m_fd", 5)) {
        chanParams.faceDetectModelPath = argv[++i];
      } else if (!::strncmp(pc,"-m_lm", 5)) {
        chanParams.landmarksModelPath = argv[++i];
      } else if (!::strncmp(pc,"-m_reid", 7)) {
        chanParams.faceRecogModelPath = argv[++i];
      } else if (!::strncmp(pc,"-t_fd", 5)) {
        chanParams.detectThreshold = atof(argv[++i]);
      } else if (!::strncmp(pc,"-t_reid", 7)) {
        chanParams.reidThreshold = atof(argv[++i]);
      } else if (!::strncmp(pc,"-fg", 3)) {
        chanParams.reidGalleryPath = argv[++i];
      } else if (!::strncmp(pc,"-min_size_fr", 12)) {
        chanParams.minSizeHW = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-di", 3)) {
        chanParams.detectInterval = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-ri", 3)) {
        chanParams.reidInterval = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-inw_fd", 7)) {
        chanParams.fdInImgWidth = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-inh_fd", 7)) {
        chanParams.fdInImgHeight = atoi(argv[++i]);
      } else if (!::strncmp(pc,"-h", 2)) {
        showUsage();
        return 0;
      } else if (!::strncmp(pc,"-i", 2)) {
        src = argv[++i];
      } else if (!::strncmp(pc,"-o", 2)) {
        dst = argv[++i];
      }  else if (!::strncmp(pc,"-V", 2)) {
        std::cout << "OVFACE " << OVFACE_VERSION << std::endl;
        return 0;
      }
    }
  }

  try {
    testLib(chanParams, src, dst);
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

bool testPair(const string & first, const string & second, int recogThreshold, bool expect, bool * ignore) 
{
  uint8_t * block = nullptr;
  int blockSize = 0;
  float detectThreshold = 0.4;

  ifstream f(first, ios::in | ios::binary | ios::ate);
  if (f.is_open()) {
    blockSize = f.tellg();
    block = new uint8_t[blockSize];
    f.seekg(0, ios::beg);
    f.read((char*)block, blockSize);
    f.close();
  }
  cv::InputArray in{ block,blockSize };
  cv::Mat firstMat = cv::imdecode(in, cv::IMREAD_COLOR);
  delete[] block;

  ifstream f2(second, ios::in | ios::binary | ios::ate);
  block = nullptr;
  blockSize = 0;
  if (f2.is_open()) {
    blockSize = f2.tellg();
    block = new uint8_t[blockSize];
    f2.seekg(0, ios::beg);
    f2.read((char*)block, blockSize);
    f2.close();
  }
  cv::InputArray in2{ block,blockSize };
  cv::Mat secondMat = cv::imdecode(in2, cv::IMREAD_COLOR);
  delete[] block;

  vector<float> recogFirst;
  vector<float> recogSecond;

  // First face.
  vector<Result> detectFirstResults = s_faceDetect->process(0, firstMat);
  if (detectFirstResults.size() > 0) {
    Result f = detectFirstResults.front();
    if (f.confidence <= detectThreshold) {
      //cout << "Seem's not a human face." << first << endl;
      *ignore = true;
      return false;
    }

    if (detectFirstResults[1].confidence >= detectThreshold) { // Too many faces
      //cout << "Too many faces." << first << endl;
      *ignore = true;
      return false;
    }

    cv::Mat cropped = getCropped(firstMat, f.location, s_faceRecog->getInferWidth(), 2);
    //cv::imwrite( "aa.jpg", cropped);
    recogFirst = s_faceRecog->process(0, cropped);
  }

  // Second face.
  vector<Result> detectSecondResults = s_faceDetect->process(0, secondMat);
  if (detectSecondResults.size() > 0) {
    Result f = detectSecondResults.front();
    if (f.confidence <= detectThreshold) {
      //cout << "Seem's not a human face." << second << endl;
      *ignore = true;
      return false;
    }

    if (detectSecondResults[1].confidence >= detectThreshold) { // Too many faces
      //cout << "Too many faces." << second << endl;
      *ignore = true;
      return false;
    }

    cv::Mat cropped = getCropped(secondMat, f.location, s_faceRecog->getInferWidth(), 2);
    //cv::imwrite( "bb.jpg", cropped);
    recogSecond = s_faceRecog->process(0, cropped);
  }

  // Compare
  cv::Mat firstVec(static_cast<int>(recogFirst.size()), 1, CV_32F);
  for (unsigned int i = 0; i < recogFirst.size(); i++) {
    firstVec.at<float>(i) = recogFirst[i];
  }
  cv::Mat secondVec(static_cast<int>(recogSecond.size()), 1, CV_32F);
  for (unsigned int i = 0; i < recogSecond.size(); i++) {
    secondVec.at<float>(i) = recogSecond[i];
  }
  float dist = computeReidDistance(firstVec, secondVec);
  /*
  if(dist>1 || dist<0){
    cout << "Error distance " << dist << " | " << first << " | " << second << endl;
  }
  */
  //cout << "Dist:"<<dist<<endl;
  bool matched = false;
  if (dist * 100 <= (100 - recogThreshold)) {
    matched = true;
  }

  return matched == expect;
}

cv::Mat getCropped(cv::Mat input, Rect rectangle, int crop_size, int style) {
  int center_x = rectangle.x + rectangle.width / 2;
  int center_y = rectangle.y + rectangle.height / 2;

  int max_crop_size = min(rectangle.width, rectangle.height);

  int adjusted = max_crop_size * 3 / 2;

  std::vector<int> good_to_crop;
  good_to_crop.push_back(adjusted / 2);
  good_to_crop.push_back(input.size().height - center_y);
  good_to_crop.push_back(input.size().width - center_x);
  good_to_crop.push_back(center_x);
  good_to_crop.push_back(center_y);

  int final_crop = *(min_element(good_to_crop.begin(), good_to_crop.end()));

  Rect pre(center_x - max_crop_size / 2, center_y - max_crop_size / 2, max_crop_size, max_crop_size);
  Rect pre2(center_x - final_crop, center_y - final_crop, final_crop * 2, final_crop * 2);

  cv::Mat r;
  if (style == 0) {
    r = input(pre);
  }
  else {
    r = input(pre2);
  }

  resize(r, r, Size(crop_size, crop_size));
  return r;
}