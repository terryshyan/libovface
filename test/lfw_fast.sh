./lfw -m_fd ../models/face-detection-retail-0005.xml \
         -m_lm ../models/landmarks-regression-retail-0009.xml \
         -m_reid ../models/model-y1-0000.xml \
         -fg ../share/faces_gallery.json \
         -t_fd 0.81 \
         -inw_fd 300 \
         -inh_fd 300
