#!/bin/bash -x

# download extra config
wget https://launchpad.net/~ubilinux/+archive/ubuntu/up/+sourcefiles/upboard-extras/0.1-1/upboard-extras_0.1.orig.tar.gz
wget https://launchpad.net/~ubilinux/+archive/ubuntu/up/+sourcefiles/upboard-extras/0.1-1/upboard-extras_0.1-1.debian.tar.xz

# upboard extra config
tar -xvf upboard-extras_0.1.orig.tar.gz
tar -xvf upboard-extras_0.1-1.debian.tar.xz

cd upboard-extras_1.0
sudo cp lib/udev/rules.d/* /lib/udev/rules.d
sudo cp etc/modules-load.d/* /etc/modules-load.d/
cd ..
cd debian
sudo chmod 777 postinst
sudo ./postinst

# gpio functionality
sudo usermod -a -G gpio ${USER}

# leds
sudo usermod -a -G leds ${USER}

# spi
sudo usermod -a -G spi ${USER}

# i2c
sudo usermod -a -G i2c ${USER}

# uart
sudo usermod -a -G dialout ${USER}

# Install rpi-gpio for UP board
wget http://ubilinux.org/ubilinux/pool/main/r/rpi-gpio/python-rpi.gpio_0.6.3+ubi1-1_amd64.deb
dpkg -i python-rpi.gpio_0.6.3+ubi1-1_amd64.deb.3
