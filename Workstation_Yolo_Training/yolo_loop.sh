#!/bin/bash

for n in $(seq 1 64) ; do
  sudo docker run --net=host -e DISPLAY=$DISPLAY --gpus all -it -d --shm-size=102400M --cpuset-cpus "0-$((n-1))" yolo_train:v3 

  container_id=$(docker ps -q | head -n 1)
  docker exec $container_id sh /root/train.sh
  
  while true; do
    docker cp "${container_id}":/stat.log "./${n}_stat.log"
    if [ $? -ne 0 ]; then
      echo "Copy log: container still running, will try again 10s later."
      sleep 10
    else
      docker cp "${container_id}":/output.log "./${n}_output.log"
      echo "Copy log: succeeded"
      break
    fi
  done

  docker stop $container_id
  docker rm $container_id
done
