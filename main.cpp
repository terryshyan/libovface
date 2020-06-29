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

void DrawObject(cv::Mat frame, cv::Rect rect, const std::string& label) {
  const cv::Scalar text_color(0, 0, 255);
  const cv::Scalar bbox_color(255, 255, 255);
  bool plot_bg = true;

  cv::rectangle(frame, rect, bbox_color);

  if (plot_bg && !label.empty()) {
    int baseLine = 0;
    const cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_size.height), cv::Point(rect.x + label_size.width, rect.y + baseLine), bbox_color, cv::FILLED);
  }

  if (!label.empty()) {
    cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1, text_color, 1, cv::LINE_AA);
  }
}

void testLib(CVAChanParams &param, const char *src, const char *dst) {
  float work_time_ms = 0.f;
  size_t work_num_frames = 0;

  //----Create chan
  VAChannel *chan =VAChannel::create(param);

  //----Capture
  cv::VideoCapture capture;
  capture.open(src);
  if (!capture.isOpened()) {
    std::cout << "[ERROR] Fail to capture video." << std::endl;
    return;
  }
  int w=capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int h=capture.get(cv::CAP_PROP_FRAME_HEIGHT);

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

