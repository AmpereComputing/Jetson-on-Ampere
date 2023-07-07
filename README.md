![Ampere Computing](https://avatars2.githubusercontent.com/u/34519842?s=400&u=1d29afaac44f477cbb0226139ec83f73faefe154&v=4)

# Jetson-on-Ampere
## Purpose of this project:
1. Demonstrate that Ampere systems provide faster compilation, and can natively compile Jetson applications. We use OpenCV as an example to demonstrate this.

2. Ampere ADLink Development Kit can be used as development platform for ARM-based devices. Users can develop application on Ampere Development Kit and directly deployed to devices like Jetson.

3. The benefit of using Ampere as the development platform is the extra CPU cores. With 64 ARM cores in total (as compared to Jetson's 12 cores), compilation speed can be speed up to more than 1.6X. Please see this tuning guide that details the performance benefits, link:

## Overview of Setup
This guide will explain the steps to install the same OpenCV application on both Jetson AGX Orin and Ampere ADLink Development Kit. Then applications compiled on Ampere Dev Kit will be copied and deployed on Jetson to show the potential of using Ampere Dev Kit as a native development platform.

To prepare for compiling and cross-compiling OpenCV, proper environment should be set up. Key requirement includes OS(Ubuntu 20.04 for ARM), GPU driver, Docker, Nvidia container toolkit, CUDA/cuDNN.
For Jetson, installing nvidia-jetpack will automatically include the first 4 components, and using NGC container will include CUDA/cuDNN. But for Ampere ADLink Development Kit, first 4 components should be installed step-by-step, and using NGC container will provide CUDA/cuDNN support. All the installation details are documented below. 

All other components and libraries, like Gstreamer, will be installed via the ```install.sh``` script.
Overview: about the containers: both has CUDA and cuDNN pre-installed. So no need to install most of the packages.

Versions of the setup in this guide:
Jetson Orin AGX 64 GB developer kit, with Jetson Linux 35.3.1. Ubuntu 20.04.6 LTS (GNU/Linux 5.10.104-tegra aarch64)
Ampere ADLink Developer Kit, with 64 Core Altra processor. Ubuntu 20.04.2 LTS (GNU/Linux 5.15.0-76-generic aarch64). With Nvidia GeForce RTX 3060, driver version 535.54.03

## Set up OpenCV environment on Jetson
### Install nvidia-jetpack after booting Jetson
```
sudo apt update
sudo apt dist-upgrade
sudo reboot
sudo apt install nvidia-jetpack
```

### Start the official Nvidia container for CUDA and cuDNN packages.
```
xhost +
# This enables the docker container to use the screen attached to Jetson, if any.

sudo docker run -it --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-cuda:11.4.19-devel
```

### Install other dependencies
Please use ```docker cp``` to copy the install.sh file in this repository to the Jetson container.
Then uncomment the lines in the install.sh file that correspond to Jetson, then execute the following lines inside the container:
```
chmod +x install.sh
sh install.sh
```
The ```install.sh``` script will install all other components and libraries, like Gstreamer. It will also download the necessary OpenCV source codes from GitHub repositories.


## Set up OpenCV environment on Ampere systems
### Requirements
Install Ubuntu 20.04 for ARM. 
If you want to compile the GPU-enabled OpenCV, an Nvidia desktop GPU and its drive installation is needed. As a side note, installation code is tested on a Nvidia GeForce 3060 GPU.

### Install Docker and Nvidia-container-toolkit
These two packages are necessary to run the official Nvidia docker image with CUDA/cuDNN.
For Jetson, these were automatically installed when installing the Jetpack components, but you need to manually install these two components when using Ampere ADLink Development Kit.

#### Install Docker on ADLink machine
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
systemctl start docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

To verify installation, run:
```
sudo docker run hello-world
docker images
```

#### Install Nvidia-container-toolkit
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

Note: For latest information and Nvidia official keyring, please see website
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker



### Start the official Nvidia container for CUDA and cuDNN packages.
```
xhost +

sudo docker run --net=host -e DISPLAY=$DISPLAY --gpus all -it nvcr.io/nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
```

Note that this container is slightly different than the one for Jetson. But it still provides the same functionality which includes CUDA and cuDNN.


### Install other dependencies
Please use ```docker cp``` to copy the install.sh file in this repository to the Jetson container.
Then uncomment the lines in the install.sh file that correspond to Jetson, then execute the following lines inside the container:
```
chmod +x install.sh
sh install.sh
```
When executing the ```install.sh``` script, it may ask for your region and city. For Pacific time, type 2 for region and 86 for city.



## Compile OpenCV on Ampere systems/Jetson
After setting up the environment, download OpenCV from GitHub and begin compilation.
```
cd ~/opencv-4.6.0/build

cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_BIN="8.7" -D CUDA_ARCH_PTX="" -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.6.0/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=ON ..
```

If you wish to compile OpenCV for CPU only, run this cmake script instead:
```
cmake -DWITH_CUDA=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=ON ..
```

Start compiling:
```
make -j36
```
For fastest compiling speed on Ampere Altra platform, use 36 cores. For Jetson AGX Orin, change this to the maximum core count, which is 12.

Note: for the variable "CUDA_ARCH_BIN", we set it to 8.7 so that the compiled OpenCV is compatible with the GPU on Jetson Orin. The compute capability for GeForce RTX 3060 is 8.6. Sample application compiled for 8.7 can be run with an 8.6 GPU, but not with lower versions like a 7.5 GPU. This is not problematic, nevertheless, as compilation is regardless of GPU compute capability. For example, a 7.5 GPU cannot run an 8.7 application, but it can compile an 8.7 application, and that application can be deployed to a 8.7 system for execution.
For complete list of GPU compute capability, please see: https://developer.nvidia.com/cuda-gpus


## Verify compilation and run sample application:
```
cd ~/opencv-4.6.0/build/bin
./opencv_version
```
This will show the OpenCV version you successfully compiled, which should be 4.6.0. For more advanced sample application run:

```
./example_cpp_facial_features ../../samples/data/lena.jpg ../../data/haarcascades/haarcascade_frontalface_alt.xml -mouth=../../data/haarcascades/haarcascade_smile.xml
```
With monitor attached, you will see OpenCV successfully recognizing the mouth and face of the person in the sample photo. Note: libcanberra error can be ignored.


## Cross compilation from Ampere to Jetson:
Copy the sample application binary from Ampere container to Jetson container, for example ```example_cpp_facial_features```. Run the same code, and you will see the same behavior when this OpenCV application is deployed to Jetson.