#!/bin/bash

sh /root/getstats.sh > /stat.log 2>&1 &
stat_pid=$!
echo $stat_pid

/usr/bin/time yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 imgsz=640 > /output.log 2>&1 &
train_pid=$!
echo $train_pid

wait $train_pid
kill $stat_pid