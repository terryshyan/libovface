./ovface -m_fd ./models/face-detection-retail-0005.xml \
         -m_lm ./models/landmarks-regression-retail-0009.xml \
         -m_reid ./models/model-y1-0000.xml \
         -fg ./share/faces_gallery.json \
         -i ./share/test.mp4 \
         -o ./share/out_v_fast.mp4 \
         -t_reid 0.664 \
         -t_fd 0.98 \
         -di 3 \
         -ri 3
