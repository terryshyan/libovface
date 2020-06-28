./ovface -m_fd ./models/face-detection-adas-0001.xml \
         -m_lm ./models/landmarks-regression-retail-0009.xml \
         -m_reid ./models/model-y1-0000.xml \
         -fg ./share/faces_gallery.json \
         -i ./share/test.mp4 \
         -out_v ./share/out_v.mp4 \
         -t_reid 0.57 -t_fd 0.6 -min_size_fr 112 \
         -greedy_reid_matching -no_show
