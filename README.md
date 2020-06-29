# libovface
A library for face recognition using openvino and insightface model.<br>

## Application scenes
face track in the video and label the face<br>
...<br>

## License
The code of libovface is released under the MIT License. 

## Development environment
CentOS 7<br>
Ubuntu<br>
Windows(TODO)<br>

## Usage
1 Install openvino<br>
Please refer https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html<br>
<br>
2 Build libovface library and test program<br>
Clone the project<br>
$git clone https://github.com/terryshyan/libovface.git<br>
$cd libovface<br>
$make<br>
<br>
3 Prepare test video file and face pictures<br>
Create a new directory (such as named res) under share directoty and copy pictures include faces to the directory<br>
Copy a video file named test.mp4 to share directory<br>
$cd libovface/share<br>
Create faces gallery file<br>
$python create_list.py res<br>
<br>
4 Run the test program<br>
$cd ..<br>
$./ovface<br>

## Reference
OpenVINO https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html <br>
InsightFace https://github.com/deepinsight/insightface <br>
