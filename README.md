![Ampere Computing](https://avatars2.githubusercontent.com/u/34519842?s=400&u=1d29afaac44f477cbb0226139ec83f73faefe154&v=4)

# Jetson Development with Ampere


## Table of Contents
* [Introduction](#purpose-of-this-project)
* [Benefit of using Ampere](#benefit-of-using-ampere)
* [Overview of setup](#overview-of-setup)
* [Set up OpenCV compilation environment on Jetson](#set-up-opencv-compilation-environment-on-jetson)
* [Set up OpenCV compilation environment on Ampere systems](#set-up-opencv-compilation-environment-on-ampere-systems)
* [Compile OpenCV on Ampere systems/Jetson](#compile-opencv-on-ampere-systems/jetson)
* [Verify compilation and run sample application](#verify-compilation-and-run-sample-application)
* [Cross compilation from Ampere to Jetson](#cross-compilation-from-ampere-to-jetson)
* [Benchmarking results](#benchmarking-results)


  * [Install Yolov8 on bare metal Jetson](#install-and-running-yolov8-on-bare-metal-jetson)
  * [Install Yolov8 in container on Jetson](#install-and-running-yolov8-on-jetson-container)
  * [Install Yolov8 in container on Workstation](#install-and-running-yolov8-on-workstation-container)

## Purpose of this project:
1. Demonstrate that Ampere systems provide fast compilation, and can natively compile Jetson applications. We use OpenCV as an example to demonstrate this.

2. [Ampere Altra Developer Platform](https://www.ipi.wiki/products/ampere-altra-developer-platform) is advantageous to developing application for ARM-based devices. You can develop application on Ampere Developer Platform and directly deploy to devices like Jetson.

Note: This is part of Ampere's [Arm Native Solutions](https://amperecomputing.com/solutions/arm-native) including cloud gaming, cloud phone, [Windows 11 on Ampere](https://github.com/AmpereComputing/Windows-11-On-Ampere), and [edge solutions](https://amperecomputing.com/home/edge). 

## Benefit of using Ampere:
The benefit of using Ampere as the developer platform is the extra CPU cores and its great energy efficiency. With 64 ARM cores in total (as compared to Jetson's 12 cores), compilation time can be speed up by more than 1.6X. Please refer to this tuning guide that details the performance benefits, link:


## Overview of setup
This guide will explain the steps to install OpenCV 4.6.0 on both NVIDIA Jetson AGX Orin™ and Ampere® Altra® Developer Platform. Then OpenCV applications compiled on Ampere will be copied and deployed to Jetson, in order to show the potential of using Ampere as an ARM native development platform.

To prepare for compiling OpenCV, proper environment should be set up on both Ampere and Jetson systems. Key requirements include OS(Ubuntu 20.04 for ARM), GPU driver, Docker, Nvidia container toolkit, CUDA/cuDNN.
For Jetson, installing nvidia-jetpack will automatically include proper OS, GPU driver, Docker and Nvidia container toolkit. Using NVIDIA NGC CUDA for L4T container on Jetson will enable CUDA/cuDNN. 
For Ampere Altra Developer Platform, most key requirements need to be installed step-by-step, and using NVIDIA NGC CUDA container will enable CUDA/cuDNN. All installation details are documented below. 

For additional components and libraries needed to compile OpenCV will be installed via the ```install.sh``` script.


### System setup
**Ampere Altra Developer Platform, with 64 Core Ampere Altra processor.**

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
Uncomment every line in ```install.sh``` that starts with “#” except the first line, then execute the following commands inside the container:
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
sudo docker run --net=host -e DISPLAY=$DISPLAY --gpus all -it nvcr.io/nvidia/cuda:11.4.19-cudnn8-devel-ubuntu20.04
```

Note that this container image is slightly different than the one for Jetson, but it still provides the same functionality which includes CUDA and cuDNN.
Make sure that the docker version matches the os version, this is very important. Also check the free memory by "free", you should have more than 2gb of memory available.
The latest version for cuda container is 12.2, but we ran 11.4 because trying to match the latest l4t-cuda container version of Jetson, for performance comparison.

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
./example_cpp_facial_features ../../samples/data/messi5.jpg ../../data/haarcascades_cuda/haarcascade_frontalface_alt.xml
```
With monitor attached, you can see the result of this sample application run. It can identify the face of the person in the input photo. Note that canberra-gtk-module error can be ignored.


## Cross compilation from Ampere to Jetson:
This additional section proves that OpenCV application compiled on Ampere system can be deployed directly to Jetson. 

To do so, find the compiled sample applications under “opencv-4.6.0/build/bin” folder on the Ampere system. Then copy any sample application binary to the same “opencv-4.6.0/build/bin” folder on Jetson system. Finally, run the copied sample application in Jetson environment. Identical results will be observed, which proves that the application can be deployed to Jetson directly. 

```
cd ~/opencv-4.6.0/build/bin
# Copy the binary from Ampere container to Jetson host:
scp ./example_cpp_facial_features dest_user@dest_ip:/
# Copy the binary from Jetson host to Jetson CUDA container:
# You can use “docker ps” to check the container_id
docker cp ./example_cpp_facial_features <container_id>:/
# Connect to the container shell, then move the binary inside the container
mv /example_cpp_facial_features ~/opencv-4.6.0/build/bin/
./example_cpp_facial_features ../../samples/data/messi5.jpg ../../data/haarcascades/haarcascade_frontalface_alt.xml
```
Note: Please make sure that ```xhost +``` is executed before starting the container.

## Benchmarking results
Please refer to official Ampere document: [Pending Link]



## Install and Running Yolov8 on bare metal Jetson
For container solution, please see the next section. [Install Yolov8 in container on Jetson](#install-and-running-yolov8-on-jetson-container)

### Install DeepStream on Jetson
Link to Nvidia official document:[^12]

Install libraries and kafka:
```
sudo apt install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev
git clone https://github.com/edenhill/librdkafka.git
cd librdkafka
git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a
./configure
make
sudo make install
sudo mkdir -p /opt/nvidia/deepstream/deepstream-6.2/lib
sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-6.2/lib
```

Install deepstream 6.2 using the Jetson tar package.
Current link to download the package:[^13]

After download, change directory to the folder with the downloaded tbz2 package:
```
sudo tar -xvf deepstream_sdk_v6.2.0_jetson.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-6.2
sudo ./install.sh
sudo ldconfig
```

### Verify deepstream installation: do this with display attached!
```
cd /opt/nvidia/deepstream/deepstream-6.2/samples/configs/deepstream-app
deepstream-app -c source30_1080p_dec_preprocess_infer-resnet_tiled_display_int8.txt 
```

### To boost the clocks on Jetson:
```
sudo nvpmodel -m 0
#It may ask you to reboot. If not, reboot Jetson with "sudo reboot now"
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Install/upgrade pip3
```
sudo apt update
sudo apt install -y python3-pip
pip3 install --upgrade pip
#Might need to update path, if so, use script below:
export PATH="/home/jetson/.local/bin:$PATH"
echo $PATH
#To check if pip3 is correctly updated:
pip3 --version
```

### Install Ultralytics Yolo and requirements
```
cd ~
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/
vi requirements.txt
```
Edit the requirement.txt file: Prepend # to the line "torch" and "torchvision", remove the # to line "onnx". We will be installing torch and torchvision manually in the next step.

```
pip3 install -r requirements.txt
```

### Install pytorch (Reference[^14])
This following line might not be needed but it is required by previous pytorch(v1.8.0)
```sudo apt-get install -y libopenblas-base libopenmpi-dev```

Install the libraries:
```
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;
```

Download the latest pytorch wheel from here[^15], latest version is 2.0.0. Alternatively, run wget:
```
wget https://developer.download.nvidia.cn/compute/redist/jp/v51/pytorch/torch-2.0.0a0+fe05266f.nv23.04-cp38-cp38-linux_aarch64.whl
```
Caution: sometimes Nvidia doesn't allow wget download, best way is to click on the link and download from browser.

Export the path to the downloaded file.(Example is in the ~ folder)
```
export TORCH_INSTALL=~/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl 
```

Install pytorch on Jetson.
```
python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export LD_LIBRARY_PATH="/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL
```

There might be error in the previous step, but could be okay. 

### Test pytorch installation (Reference[^15])
```
python3 -i
>>> import torch
>>> print(torch.__version__)
>>> print('CUDA available: ' + str(torch.cuda.is_available()))
>>> print('cuDNN version: ' + str(torch.backends.cudnn.version()))
>>> a = torch.cuda.FloatTensor(2).zero_()
>>> print('Tensor a = ' + str(a))
>>> b = torch.randn(2).cuda()
>>> print('Tensor b = ' + str(b))
>>> c = a + b
>>> print('Tensor c = ' + str(c))
```

### Install torchvision (Reference[^15])
```
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
```

Please see the reference website for compatibility matrix. For pytorch version 2.0.0, torchvision 0.15.1 is compatible.
```
git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision
cd torchvision/
export BUILD_VERSION=0.15.1
python3 setup.py install --user
```

To test torchvision you must exit the installation directory:
```
cd ..
python3 -i
>>> import torchvision
>>> print(torchvision.__version__)
```

### Pull deepstream-yolo for Yolov8 test run and set configs (Reference[^16]):
```
cd ~
git clone https://github.com/marcoslucianops/DeepStream-Yolo
cp DeepStream-Yolo/utils/export_yoloV8.py ultralytics/
cd ~/ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

Attempt to install onnxsim and onnxruntime:
```
sudo apt-get install cmake
pip3 install onnxsim onnxruntime
```
onnxsim and onnxruntime installation have problems, but this is okay. We will skip using the --simplify flag when running ```export_yoloV8.py```.
```
python3 export_yoloV8.py -w yolov8s.pt
```

Note: To change the inference size (defaut: 640)
```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Run Make.
```
cp yolov8s.onnx ../DeepStream-Yolo/
cd ../DeepStream-Yolo/
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
```

If want to change the detected classes(default is 80):
```
vi config_infer_primary_yoloV8.txt 
```

Configurate deepstream run:
```
vi deepstream_app_config.txt
```
Make sure the config-file is set:
```
[primary-gie]
...
config-file=config_infer_primary_yoloV8.txt
#Also if you want to loop the video infinitely:
[tests]
file-loop=1
```
You can also configure the input video and resolution of the run. For more information:[^17] [^18] [^19]
For multistream configuration, see [^21]

### To run Yolov8 (must have a monitor attached):
```
deepstream-app -c deepstream_app_config.txt
```
You will see a Yolo application pop up with object detection. Congrats!

## Install and Running Yolov8 on Workstation container
[Please refer to this](https://github.com/AmpereComputing/NVIDIA-GPU-Accelerated-Linux-Desktop-on-Ampere/tree/issueRobby#install-and-running-yolov8-on-workstation-container)

## Install and Running Yolov8 on Jetson container
### Start the NGC DeepStream container on Jetson(Reference[^20]):

We use the DeepStream-l4t-Triton container as the base container to build our environment
Please first make sure there is a display connected to your Jetson device.
```
xhost +
sudo docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.2 -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/deepstream-l4t:6.2-triton
```
Make sure you used the tag <6.2-triton> for the container.
Other tags like the base DeepStream-l4t container is not supported.

Your terminal should now enter the docker bash. 

### Build the environment for YoloV8 in DeepStream-l4t container:
In the container shell, please execute this following command to install other libraries.
```
/opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

Then please follow the same steps in the section 
TODO: "Install and Running Yolov8 (on bare metal Jetson/Workstation)" above, from section "Install/upgrade pip3" to "Pull deepstream-yolo for Yolov8 test run and set configs", to set up YoloV8 running environment. The steps are mostly the same, except for some differences noted below:

**Different steps:**
1.Remove "sudo" for any installation script, the docker container is already running with root permissions

2.When update pip3 as shown above, there's no need to change the path variable.

3.When editing the requirements.txt file(i.e. ```vi requirements.txt```) before installing yolo requirements, only comment out the two lines of torch and torchvision. Do not uncomment onnx, it is already installed in this container.

4.When downloading the pytorch wheel does not work, download the file to Jetson first, then use docker cp to copy into the container. For example:
First find the DeepStream container id using ```docker ps```. Then copy the downloaded pytorch wheel file into the deepstream container(replace the "path/to" to the actual path to the file on Jetson host, and the <container ID> with the Jetson deepstream container ID):
```
docker cp /path/to/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl <container ID>:/root
```

### Running YoloV8 in deepstream container (must have a monitor attached):
After the environment is set in the container, use the same command inside the ```~/DeepStream``` folder to start the DeepStream app.
```
deepstream-app -c deepstream_app_config.txt
```

### To commit this container:
After everything is installed and Deepstream running successfully, you can choose to commit this container, so next time this committed image can be used to start running YoloV8 with deepstream.
```
docker commit <container ID> <repository>:<tag>
```

To start a commited container:
```
xhost +
sudo docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.2 -v /tmp/.X11-unix/:/tmp/.X11-unix <repository>:<tag>
```

## Training a YoloV8 model on Workstation
TODO: finish step "Install and runnin Yolov8 on Workstation container"

For the last step of installing YOLO's required packages:
Alternatively, if you want to specify the versions to match the package versions on Jetson:
Use the requirements.txt file in the appendix.
```
pip install -r requirements.txt
pip install ultralytics
```

### Then build a container that can be reused in the training process
All relevant codes are in the ```Workstation_Yolo_Training``` folder of this repository. Use ```docker cp``` command to copy this folder from the workstation host into the container.

Then execute install.sh file inside the container.

Then put these two files in the home folder(/root) inside the container: ```train.sh```, ```getstats.sh```.

### Then open another terminal connecting to the ADLink host, but not this container. Commit this docker image, name it as ```yolocontainer:v1```
TODO:commit this container.

Use the script ```yolo_loop.sh``` in Workstation host to start YOLO containers with 1, 2 ...64 cores, and get running statistics.
```sudo sh yolo_loop.sh```

After executing the script, the running statistics will be saved in the folder of the ```yolo_loop.sh``` script. To interpret the results, for example, 10_output.log will be the /usr/bin/time output for YOLO training in a container with 10 cpu cores enabled. 10_stat.log will be the CPU, GPU usage details captured when training in a container with 10 cpu cores.





## Training a Yolov8 model on Jetson container
#TODO: don't forget to boost the clocks on Jetson.

### Start the NGC DeepStream container on Jetson(Reference[^20]):
We use the DeepStream-l4t-Triton container as the base container to build our environment
```
xhost +
sudo docker run -it --net=host --runtime nvidia -e DISPLAY=$DISPLAY --shm-size=102400M -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/deepstream-l4t:6.2-triton
```
Note that for shm-size, input a memory large enough to let the YOLO model train (you will know an shm-size is not enough if yolo training crashes).

### TODO: Then follow the steps previously to install torch and torchvision in the container.

### Finally, install YOLO
pip install ultralytics

### Benchmark YOLO training time on Jetson, with CPU core count as variable:
All relevant codes are in the ```Jetson_Yolo_training``` folder of this repository. Use ```docker cp``` command to copy this folder from the workstation host into the container.

We actually don't build a container, because the space is limited on Jetson. Instead we include a script ```install.sh``` to install all the YOLO dependencies when a new container is started. Also the yolo_loop.sh script will automatically copy the install.sh and train.sh into the container. And also getstats.sh here is used outside the container, because you cannot use "tegrastats" inside a Jetson container.

Create a folder containing all these four files below on Jetson host(not container). Then run:
```sudo sh yolo_loop.sh```
Outputs will be generated to the same folder.


## Software versions when benchmarking YOLO training
To show the version of all YOLO related packages, do:
```pip install ultralytics```

Versions used in our training: 

Same across both systems:
numpy: 1.24.1
kiwisolver: 1.4.4
pandas: 2.0.2
opencv-python: 4.7.0.72
pillow: 9.4.0
pyyaml: 6.0
requests: 2.31.0
scipy: 1.10.1
pytorch: 2.0.0
torchvision: 0.15.1

Differences:
ADLink: Python 3.10.6 (main, May 29 2023, 11:10:38) [GCC 11.3.0] on linux
Jetson: Python 3.8.10 (default, May 26 2023, 14:05:08)
Python version is already latest on Jetson, thus cannot update it to match 3.10

CUDA version and cuda compiler: 12.1(ADLink) vs 11.4(Jetson)
GCC version 11 vs 9


## Convenient script to get training time from output logs and stat logs:
for n in $(seq 1 12) ; do
  tail -2 ${n}_output.log | xargs | cut -d' ' -f3
done

for n in $(seq 1 12) ; do
  tail -1 ${n}_stat.log | xargs | cut -d',' -f2
done







## References
[^1]: 
[^2]: 
[^3]: 
[^4]: 
[^5]: 
[^6]: 
[^7]: https://forums.developer.nvidia.com/t/tutorial-using-sdkmanager-for-flashing-on-windows-via-wsl2-wslg/225759
[^8]: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jetson-linux-flash-x86
[^9]: https://developer.nvidia.com/embedded/jetson-linux
[^10]: https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_5.0.2/Flashing_Board
[^11]: https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3251/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/quick_start.html#wwpID0EAAMNHA
[^12]: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html
[^13]: https://developer.nvidia.com/downloads/deepstream-sdk-v620-jetson-tbz2
[^14]: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
[^15]: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
[^16]: https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/YOLOv8.md
[^17]: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html#configuration-groups
[^18]: https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/cus
[^19]: https://maouriyan.medium.com/the-friendly-guide-to-build-deepstream-application-3e78cb36d9f2
[^20]: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream-l4t
[^21]: https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/#multistream-configuration
[^22]: 
[^23]: 
