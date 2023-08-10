
#!/bin/bash
/usr/bin/time yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 imgsz=640 > /output.log 2>&1