#!/bin/bash

pip install ultralytics
yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 imgsz=640
apt-get update
apt-get install time apcalc sysstat -y