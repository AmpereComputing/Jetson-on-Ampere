#!/bin/bash
cd ~
# Jetson: can skip these 4 lines below
apt-get update
apt-get -y install software-properties-common
apt-add-repository universe -y
apt-get update

apt-get -y install build-essential make cmake libgtk2.0-dev pkg-config cmake-curses-gui
apt-get -y install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
apt-get -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
apt-get -y install libv4l-dev
apt-get -y install libeigen3-dev
apt-get -y install libglew-dev
apt-get -y install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev
# Jetson: apt-get -y install libdc1394-22-dev
apt-get -y install git wget unzip
apt-get install -y python3.8-dev python3-numpy
# Jetson: apt-get -y install python-dev python-numpy
apt-get install -y libv4l-dev v4l-utils qv4l2
# Jetson: apt-get -y install v4l2ucp

wget https://github.com/opencv/opencv/archive/4.6.0.zip
unzip 4.6.0.zip
rm 4.6.0.zip
wget https://github.com/opencv/opencv_contrib/archive/4.6.0.zip
unzip 4.6.0.zip
rm 4.6.0.zip
mkdir ~/opencv-4.6.0/build