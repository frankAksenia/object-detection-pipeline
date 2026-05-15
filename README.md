# MLOps VisDrone YOLO Pipeline

End-to-end computer vision MLOps pipeline for real-time object detection on the **VisDrone2019-DET** dataset using **YOLO**, **Google Colab GPU training**, **dataset conversion**, **evaluation artifacts**, and a production-oriented project structure.

This project includes:

- Dataset download and preparation
- VisDrone-to-YOLO annotation conversion
- GPU-based YOLO training in Google Colab
- Model evaluation with mAP, precision, recall, confusion matrix, and PR curves
- Prediction visualization
- Model export readiness
- Clean source-code structure for API, Docker, and deployment work

---

## Project Overview

The goal is to build a reproducible object detection pipeline for drone imagery.

The current pipeline trains a YOLO model to detect the following classes:

```text
0: pedestrian
1: people
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor
```
