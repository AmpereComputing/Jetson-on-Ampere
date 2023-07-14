![Ampere Computing](https://avatars2.githubusercontent.com/u/34519842?s=400&u=1d29afaac44f477cbb0226139ec83f73faefe154&v=4)

# Jetson-on-Ampere
## Purpose of this project:
1. Demonstrate that Ampere systems provide faster compilation, and can natively compile Jetson applications. We use OpenCV as an example to demonstrate this.

2. Ampere Altra Developer Platform is advantageous to developing application for ARM-based devices. You can develop application on Ampere Developer Platform and directly deploy to devices like Jetson.


## Benefit of using Ampere:
The benefit of using Ampere as the developer platform is the extra CPU cores and its great energy efficiency. With 64 ARM cores in total (as compared to Jetson's 12 cores), compilation time can be speed up by more than 1.6X. Please refer to this tuning guide that details the performance benefits, link:


## Overview of Setup
This guide will explain the steps to install OpenCV 4.6.0 on both NVIDIA Jetson AGX Orin™ and Ampere® Altra® Developer Platform. Then OpenCV applications compiled on Ampere will be copied and deployed to Jetson, in order to show the potential of using Ampere as an ARM native development platform.

To prepare for compiling OpenCV, proper environment should be set up on both Ampere and Jetson systems. Key requirements include OS(Ubuntu 20.04 for ARM), GPU driver, Docker, Nvidia container toolkit, CUDA/cuDNN.
For Jetson, installing nvidia-jetpack will automatically include proper OS, GPU driver, Docker and Nvidia container toolkit. Using NVIDIA NGC CUDA for L4T container on Jetson will enable CUDA/cuDNN. 
For Ampere ADLink Development Kit, most key requirements need to be installed step-by-step, and using NVIDIA NGC CUDA container will enable CUDA/cuDNN. All installation details are documented below. 

For additional components and libraries needed to compile OpenCV will be installed via the ```install.sh``` script.


### System setup
**ADLink Ampere Altra Developer Platform, with 64 Core Ampere Altra processor.**

•	Ubuntu 20.04.2 LTS (GNU/Linux 5.15.0-76-generic aarch64).

•	GPU config: NVIDIA GeForce RTX 3060

•	GPU driver: 535.54.03

**Jetson Orin AGX 64 GB developer kit**

•	OS and Kernel: Ubuntu 20.04.6 LTS (GNU/Linux 5.10.104-tegra aarch64)

•	GPU config: NVIDIA Ampere architecture with 2048 NVIDIA® CUDA® cores and 64 Tensor cores

•	GPU driver: default (Jetson Linux 35.3.1) 


## Set up OpenCV compilation environment on Jetson
### Install nvidia-jetpack after booting Jetson
```
sudo apt update
sudo apt dist-upgrade
sudo reboot
sudo apt install nvidia-jetpack
```

### Start the Nvidia NGC container with CUDA and cuDNN packages.
```
xhost +
# This enables the container to use screens attached to Jetson, if any.
sudo docker run -it --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-cuda:11.4.19-devel
```

### Install other dependencies
Connect to the docker container, install git, then fetch this GitHub repository. You should be able to download the file ```install.sh```. Alternatively you can download files to Jetson host, then use ```docker cp``` to copy the ```install.sh``` file into the Jetson container.
Then uncomment the lines in the ```install.sh``` script that correspond to Jetson, and execute the following commands inside the container:
```
chmod +x install.sh
sh install.sh
```
The ```install.sh``` script will install all other packages and libraries, such as Gstreamer, Make. It will also download the necessary OpenCV source code from GitHub repositories.


## Set up OpenCV compilation environment on Ampere systems
### Requirements
Install Ubuntu 20.04 for ARM. 
If you want to compile OpenCV with GPU enabled, you need to have an Nvidia desktop GPU and install its driver. For reference, all GPU-related code in this guide is executed on a Nvidia GeForce RTX 3060 GPU.

### Install Docker and NVIDIA Container Toolkit
These two packages are necessary to run the official Nvidia NGC docker image with CUDA/cuDNN.
For Jetson, these were installed when you install the Jetpack components. For Ampere Developer Platform, follow the steps below to install these two packages.

#### Install Docker
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
systemctl start docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

To verify installation, execute the following commands.
```
sudo docker run hello-world
docker images
```

#### Install NVIDIA Container Toolkit
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Note: NVIDIA occasionally changes official keyring. If key error occurs, please refer to website:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker


### Start the official Nvidia container for CUDA and cuDNN packages.
```
xhost +
sudo docker run --net=host -e DISPLAY=$DISPLAY --gpus all -it nvcr.io/nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
```

Note that this container image is slightly different than the one for Jetson, but it still provides the same functionality which includes CUDA and cuDNN.


### Install other dependencies
In host terminal, use ```docker cp``` to copy the ```install.sh``` file in this repository to the Jetson container.
Then connect to the container and execute the following commands to install additional packages.
```
chmod +x install.sh
sh install.sh
```
During execution of the ```install.sh``` script, the terminal might ask for your region and city. For Pacific time zone, type 2 for region and 86 for city.


## Compile OpenCV on Ampere systems/Jetson
After running ```install.sh```, change directory to the OpenCV 4.6.0 source code folder and start CMake.
```
cd ~/opencv-4.6.0/build
cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_BIN="8.7" -D CUDA_ARCH_PTX="" -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.6.0/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=ON ..
```

If you wish to compile OpenCV for CPU only, run this CMake script instead:
```
cmake -DWITH_CUDA=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=ON ..
```

To compile the OpenCV library using 36 CPU cores:
```
make -j36
```
Use 36 cores for optimal compilation on Ampere Altra platform, and use max cores for best performance. For Jetson AGX Orin, use its maximum core count which is 12.

Note: The OpenCV CMake configuration option “CUDA_ARCH_BIN” is set to 8.7 so that the compiled OpenCV application is matching the compute capability of the Jetson AGX Orin GPU (8.7). 
The Ampere Altra Development Platform in this guide has an NVIDIA GeForce RTX 3060 GPU, with a compute capability of 8.6. Upon testing, OpenCV sample application compiled with CUDA_ARCH_BIN=“8.7” can run on a system with GPU compute capability 8.6, but not with lower GPU compute capability like 7.5. Therefore, the best practice is to set "CUDA_ARCH_BIN" to the compute capability of the Jetson device. This will ensure that compiled sample application can always be deployed to an 8.7 Jetson system for execution. 
For a complete list of GPU compute capability, please refer to: https://developer.nvidia.com/cuda-gpus



## Verify compilation and run sample application:
```
cd ~/opencv-4.6.0/build/bin
./opencv_version
```
This will output the version of compiled OpenCV, which should be 4.6.0. 

You can run a more advanced sample application, for example:
```
./example_cpp_facial_features ../../samples/data/lena.jpg ../../data/haarcascades/haarcascade_frontalface_alt.xml -mouth=../../data/haarcascades/haarcascade_smile.xml
```
With monitor attached, you can see the result of this sample application run. It can identify the face and mouth of the person in the input photo. Note that canberra-gtk-module error can be ignored.


## Cross compilation from Ampere to Jetson:
After compiling OpenCV on Ampere system, sample application binaries can be copied and run on Jetson system without any issue. 
Copy the sample application binary from your Ampere container to the same location in your Jetson container, for example ```opencv-4.6.0/build/bin/example_cpp_facial_features```. Finally, run the copied sample application in Jetson environment. Identical result will be observed, which proves that the application can be deployed to Jetson directly. 