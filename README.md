# libovface
A library for face recognition using openvino and insightface model.

## License
The code of libovface is released under the MIT License. 

## Development environment
CentOS 7

## Usage
1 Install openvino<br>
Please refer https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html<br>
<br>
2 Build libovface library and test program<br>
$cd libovface<br>
$make<br>
<br>
3 prepare test video file and pictures<br>
crate a new directory under share directoty and copy pictures incude faces to directory,such as res<br>
copy a video file named test.mp4 to share directory<br>
$cd libovface/share<br>
$python create_list.py res<br>
<br>
4 Run the test program<br>
$cd ..<br>
$./ovface<br>
