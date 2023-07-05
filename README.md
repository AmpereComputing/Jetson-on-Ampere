# Jetson-on-Ampere


## Compile OpenCV on Jetson
```
xhost +

sudo docker run -it --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-cuda:11.4.19-devel
```

Uncomment the lines in the install.sh file that correspond to Jetson, then execute:
```
sh install.sh
```


## Compile OpenCV on Ampere systems
**Requirements:**
Ubuntu 20.04 for ARM. Docker, Nvidia-container-toolkit. Nvidia desktop GPU and driver.
This is tested on a Nvidia GeForce 3060 GPU.

```
xhost +

sudo docker run --net=host -e DISPLAY=$DISPLAY --gpus all -it nvcr.io/nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

sh install.sh
```

## Compile OpenCV on Ampere systems/Jetson
```
wget https://github.com/opencv/opencv/archive/4.6.0.zip
unzip 4.6.0.zip
rm 4.6.0.zip
wget https://github.com/opencv/opencv_contrib/archive/4.6.0.zip
unzip 4.6.0.zip
rm 4.6.0.zip
mkdir ~/opencv-4.6.0/build
cd ~/opencv-4.6.0/build

cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_BIN="8.7" -D CUDA_ARCH_PTX="" -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.6.0/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=ON ..

# For fastest compiling speed on Ampere Altra platform, use 36 cores. For Jetson AGX Orin, change this to the maximum core count, which is 12.
make -j36
```