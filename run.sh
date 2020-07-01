./ovface -m_fd ./models/face-detection-retail-0005.xml \
         -m_lm ./models/landmarks-regression-retail-0009.xml \
         -m_reid ./models/face-reidentification-retail-0095.xml \
         -fg ./share/faces_gallery.json \
         -i ./share/test.mp4 \
         -o ./share/out_openvino.mp4 \
         -t_reid 0.57 \
         -t_fd 0.98 \
         -di 3 \
         -ri 3
