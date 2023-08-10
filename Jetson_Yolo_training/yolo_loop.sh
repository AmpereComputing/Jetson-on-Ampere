#!/bin/bash

for n in $(seq 1 12) ; do
  sudo docker run --net=host -e DISPLAY=$DISPLAY --runtime nvidia -it -d --shm-size=102400M --cpuset-cpus "0-$((n-1))" yolocontainer:v1

  container_id=$(docker ps -q | head -n 1)
  # add
  docker cp install.sh "${container_id}":/root/install.sh
  docker cp train.sh "${container_id}":/root/train.sh
  docker exec $container_id sh /root/install.sh > /dev/null 2>&1
  
  sh getstats.sh > "${n}_stat.log" 2>&1 &
  stat_pid=$!
  
  docker exec $container_id sh /root/train.sh

  kill $stat_pid

  docker cp "${container_id}":/output.log ./${n}_output.log
  echo "Successfully copied yolo output"

  docker stop $container_id
  docker rm $container_id
done
