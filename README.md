# CHART: Classifier for Human Activities in Real-Time



Subfolders named "Dataset_X" contain code to generate
training/validation/test csv files containing MoveNet
coordinates for the respective datasets.

The "Deep Learning" subfolder contains Jupyter Notebooks
to train the various models and some Fisher Ratio utilities

The "InsightFace_Testing" subfolder contains (i) a Streamlit
app to test InsightFace face detection with FPS, and
(ii) code to test InsightFace with blurring on a given video
(it will save the processed video)

The "insightface_vs_proposed" subfolder contains code for
(i) OpenCV FaceCascade, (ii) InsightFace face detection,
(iii) Proposed face detection method in order to compare
them through a proposed overlap metric (overlap_metrics.py)

The "MoveNet_Testing" subfolder contains code to test the
classifier system on images (movenet_testing_image.py) and
also on a video (movenet_testing_video.py)

The "WebApp" subfolder contains the necessary code to run and
deploy a Streamlit web application for the human activity
recognition system

For any code using a Streamlit web application, use the
app_envm.yml file for the conda environment. For the
notebooks in "Deep Learning" subfolder, run these in
Google Colab. For other non-app code, use the
non_app_envm.yml file for the environment.


# Citation
```bibtex
@inproceedings{singh2023,
author = {Singh, Ishneet Sukhvinder and Kaza, Pradyoth and Hosler Iv, Peter Gregory and Chin, Zheng Yang and Ang, Kai Keng},
title = {Real-Time Privacy Preserving Human Activity Recognition on Mobile using 1DCNN-BiLSTM Deep Learning},
year = {2023},
booktitle = {Proceedings of the 2023 5th International Conference on Image, Video and Signal Processing},
}
```

