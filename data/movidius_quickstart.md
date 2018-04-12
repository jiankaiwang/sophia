# Movidius Neural Compute Stick







*   **Reference**
    *   Developer: https://developer.movidius.com/start
    *   Document: https://movidius.github.io/ncsdk
    *   github: https://github.com/movidius/ncsdk





## Quick Start



### 作業系統更新



*   Raspberry Pi  需更新至 stretch 9
    *   https://linuxconfig.org/raspbian-gnu-linux-upgrade-from-jessie-to-raspbian-stretch-9
*   Ubuntu: `sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get autoremove -y`




### 安裝 NCS SDK

```bash
$ cd ~
$ mkdir -p ./workspace
$ cd ./workspace
$ git clone https://github.com/movidius/ncsdk.git
$ cd ./ncsdk
$ make install

# source the bashrc to pickup new $PYTHONPATH
# after source, the python version is replaced with caffe-based python
$ source ~/.bashrc

# it shows :/opt/movidius/caffe/python by default
$ echo $PYTHONPATH

# use make help to show further commands
$ make help
```



### NCS 範例



[**Source Code**] 安裝 OpenCV 於 ARM, Raspberry Pi 等。

[ys7yoo/PiOpenCV](github.com/ys7yoo/PiOpenCV)



[**Ubuntu**] 安裝 Tensorflow==1.4。

```bash
$ wget https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
$ pip3 install tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
```



[**Raspberry Pi 3**] 安裝 Tensorflow==1.4。

```bash
$ wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.4.0/tensorflow-1.4.0-cp35-none-linux_armv7l.whl
$ sudo pip3 install tensorflow-1.4.0-cp35-none-linux_armv7l.whl
```



**Notice**: 將 Movidius 連接裝置。

```bash
$ cd ~/workspace/ncsdk

# install examples
# make sure you have already installed tensorflow
$ make examples

# start an example
$ cd examples/app/hello_ncs_py/
# 
# making run
# python3 hello_ncs.py;
# Hello NCS! Device opened normally.
# Goodbye NCS! Device closed normally.
# NCS device working.
#
$ make run

# view the source code
$ vim ./hello_ncs.py
```



### NCS 與 DL 框架範例



**Caffe** 範例

```bash
# example for caffe with googlenet
$ cd examples/caffe
$ cd GoogLeNet

# make check to look for 'Result: Validation Pass'
$ make check

# make profile to run the sdk profile
# and generate the output_report.html
$ make profile
$ chromium-browser ~/workspace/ncsdk/examples/caffe/GoogLeNet/output_report.html

# make help to check the command
$ make help
$ make cpp

# make py
# the top inference results are output
$ make run_py

# make cpp
# the top inference results are output
$ make run_cpp
```



**Tensorflow** 範例

```bash
# example for tensorflow
$ cd examples/tensorflow
$ cd inception_v3

# check for 'Result: Validation Pass'
$ make check

# make profile to run the sdk profile
# and generate the output_report.html
$ make profile

# run the example
$ make all
$ python run.py
```















