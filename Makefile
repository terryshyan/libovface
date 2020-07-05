OPENVINO_PATH := /opt/intel/openvino
CXX_FLAGS = -std=c++11  -Wuninitialized -Winit-self -Wmaybe-uninitialized -Wall -g -DNDEBUG \
			-I./include -I./common \
			-I$(OPENVINO_PATH)/opencv/include \
			-I$(OPENVINO_PATH)/deployment_tools/inference_engine/include \
			-I$(OPENVINO_PATH)/deployment_tools/ngraph/include  \
			-Wno-error=deprecated-declarations -DGFLAGS_IS_A_DLL=0 -DNGRAPH_JSON_DISABLE -DNGRAPH_VERSION=\"0.29.0-rc.0+edc65ca\" -DUSE_OPENCV

LDFLAGS:= 

LIBS := -rdynamic $(OPENVINO_PATH)/inference_engine/lib/intel64/libinference_engine.so \
				  $(OPENVINO_PATH)/inference_engine/lib/intel64/libinference_engine_legacy.so \
				  $(OPENVINO_PATH)/deployment_tools/ngraph/lib/libngraph.so \
				  $(OPENVINO_PATH)/opencv/lib/libopencv_highgui.so.4.3.0 -ldl -lpthread \
				  $(OPENVINO_PATH)/opencv/lib/libopencv_videoio.so.4.3.0 \
				  $(OPENVINO_PATH)/opencv/lib/libopencv_imgcodecs.so.4.3.0 \
				  $(OPENVINO_PATH)/opencv/lib/libopencv_imgproc.so.4.3.0 \
				  $(OPENVINO_PATH)/opencv/lib/libopencv_core.so.4.3.0

SRC_PATH := src

OBJS = $(SRC_PATH)/align_transform.o $(SRC_PATH)/cnn.o $(SRC_PATH)/detector.o $(SRC_PATH)/recognizer.o $(SRC_PATH)/tracker.o \
       $(SRC_PATH)/image_grabber.o $(SRC_PATH)/logger.o  $(SRC_PATH)/reid_gallery.o $(SRC_PATH)/ovface_impl.o
TARGET = libovface.so
PROGS = ovface

CXX = /usr/bin/c++
​COMPILE = $(CXX) $(CXX_FLAGS)
LINK = $(CXX)

.PHONY: all debug clean $(TARGET) $(PROGS)

all: $(TARGET) $(PROGS)

debug:
	$(MAKE) all DEBUG=-g3 MODSTRIP=

$(TARGET): $(OBJS)
	$(LINK) $(OBJS) -shared -fPIC -o $@ $(LDFLAGS) $(LIBS)

$(PROGS):$(TARGET)
	$(​COMPILE) main.cpp -o $@ $(TARGET) $(LDFLAGS) $(LIBS)

$(SRC_PATH)/%.o: $(SRC_PATH)/%.cpp  
	$(​COMPILE) -fPIC -c $< -o $@
  
clean:  
	rm -rf $(OBJS) $(TARGET) $(PROGS)
