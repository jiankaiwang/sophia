# Movidius Neural Compute Stick



*   **Reference**
    *   Developer: https://developer.movidius.com/start
    *   Document: https://movidius.github.io/ncsdk
    *   github: https://github.com/movidius/ncsdk
*   https://linuxconfig.org/raspbian-gnu-linux-upgrade-from-jessie-to-raspbian-stretch-9



## Quick Start



### 安裝 NC SDK

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



### NC 範例



**Notice**: 將 Movidius 連接裝置。

```bash
$ cd ~/workspace/ncsdk

# install examples
$ make examples

$ cd examples

```



















