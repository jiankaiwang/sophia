Jetson TX2
==========

Preparation
-----------

-   OS and CUDA Installation:
    https://jkjung-avt.github.io/opencv3-on-tx2/

-   OpenCV Installation: https://jkjung-avt.github.io/opencv3-on-tx2/

    -   solve the **gl.h** issue

    ``` {.cpp}
    # in /usr/local/cuda-9.0/include/cuda_gl_interop.h 

    /* comment the following sections
    #if defined(__arm__) || defined(__aarch64__)
    #ifndef GL_VERSION
    #error Please include the appropriate gl headers before including cuda_gl_interop.h
    #endif
    #else
    #include <GL/gl.h>
    #endif
    */

    // add the resource directly
    #include <GL/gl.h>
    ```

-   **TensorFlow** Installation:

Dependency Installation:

``` {.bash}
#install dependencies
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer -y
sudo apt-get install zip unzip autoconf automake libtool curl zlib1g-dev maven -y
sudo apt install python3-numpy python3-dev python3-pip python3-wheel
```

Installing Bazel:

``` {.bash}
bazel_version=0.10.0
wget https://github.com/bazelbuild/bazel/releases/download/$bazel_version/bazel-$bazel_version-dist.zip
unzip bazel-$bazel_version-dist.zip -d bazel-dist
sudo chmod -R ug+rwx bazel-dist
cd bazel-dist
./compile.sh 
sudo cp output/bazel /usr/local/bin
```

Installing Tensorflow:

``` {.bash}
git clone --recursive https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v1.7.0
```

Edit the environment setting (for this installarion only):

``` {.bash}
export PYTHON_BIN_PATH=$(which python3)

# No Google Cloud Platform support
export TF_NEED_GCP=0

# No Hadoop file system support
export TF_NEED_HDFS=0

# Use CUDA
export TF_NEED_CUDA=1

# Setup gcc ; just use the default
export GCC_HOST_COMPILER_PATH=$(which gcc)

# TF CUDA Version 
export TF_CUDA_VERSION=9.0

# CUDA path
export CUDA_TOOLKIT_PATH=/usr/local/cuda

# cuDNN
export TF_CUDNN_VERSION=7
export CUDNN_INSTALL_PATH=/usr/lib/aarch64-linux-gnu

# CUDA compute capability
export TF_CUDA_COMPUTE_CAPABILITIES=6.2
export CC_OPT_FLAGS=-march=native
export TF_NEED_JEMALLOC=1
export TF_NEED_OPENCL=0
export TF_ENABLE_XLA=1
```

Configure the tensorflow installation:

``` {.bash}
./configure
```

Compile the resource:

``` {.bash}
bazel build -c opt --verbose_failures --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

Package the installation:

``` {.bash}
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

Move the wheel file to the directory for installation:

``` {.bash}
mv /tmp/tensorflow_pkg/tensorflow-1.7.0*-linux_aarch64.whl ../
```

Install the wheel file:

``` {.bash}
cd ..
sudo pip install tensorflow-1.7.0-cp35-cp35m-linux_aarch64.whl
```

Test the installation:

``` {.bash}
vim test_tensorflow.py
```

``` {.python}
#!/usr/bin/env python 

import tensorflow as tf
hello = tf.constant('Hello world, TensorFlow on TX2!')
sess = tf.Session()
print(sess.run(hello))
```

``` {.bash}
python test_tensorflow.py
```

Reference:

-   Pre-Built: https://github.com/jetsonhacks/installTensorFlowJetsonTX
-   Source: https://github.com/jetsonhacks/installTensorFlowTX2
