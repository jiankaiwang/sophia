### 深度學習的流程

* **Data Reading**: 讀入資料
* **Data Preprocessing**: 進行資料前處理
* **Create a Model**: 定義神經網路及超參數
* **Learning the Model Parameters**: 學習過程
* **Evaluating the Model**: 驗證模型

### 神經網路組成

* 活化函式: Sigmoid, tanh, ReLu, Softplus
* 多層神經網路
* 輸出函式: 恆等式, Softmax 

### 神經網路學習

* 損失函數: Mean Square Error, Cross Entropy Error
* 微分與偏微分
* 求梯度: 梯度下降 (Gradient Descent), 梯度下降與學習

* 使用正向傳播梯度下降法訓練的神經網路
* 計算權重的梯度
* 學習過程

* 導入批次的學習

### 反向傳播法

* 多層架構: 正向傳播 vs. 反向傳播
* 活化函式的正向與反向傳播
* 全連接層的正向與反向傳播
* 輸出函式的正向與反向傳播
* 梯度確認
* 反向傳播的學習
* 反向傳播的損失度

### 神經網路進階學習

* 最佳優化方法: Stochastic Gradient Descent, Momentum, AdaGrad, Adam, nesterov, RMSprop
* 權重參數的預設值: Xavier method, Kaiming He method
* Batch Normalization
* Overfitting 與 Normalization: Weight Decay / Regularization, Dropout
* 超參數的設定: Validation data, 尋找最佳超參數的方法 

### 深度學習框架介紹

* Keras: [Keras for Python](data/Keras_Quickstart.html)
* Tensorflow: [Tensorflow for Python](data/Tensorflow_Quickstart_Python.html)
    * [Tensorflow architecture](data/Basic_Tensorflow.html): API, Variable Scope vs. Name Scope, Session, CPU/GPU
    * batch normalization, dropout, regularization
    * neural network function
        - [activation function](data/BasicLearning_Tensorflow.html)
        - optimization function
    * model portability: `ckpt with meta`, `frozen pb`
    * queue and threading
    * TF-Slim

### 深度學習問題與主流方法

* 基礎神經網路
  * 監督式學習網路 (Supervised Learning Network)
      * MLP (Multi-Layer Preceptron)
      * CNN (Convolutional Neural Network): [Basic CNN](data/BasicCNN_Tensorflow.html), [CNN Tensorboard](data/CNN_Tensorboard.html)
        * Convolutional Layer, Filter / Mask / Kernel, Padding, Stride, Pooling Layer
        * im2col / col2im
      * RNN (Recurrent Neural Network)
  * 非監督式學習網路 (Unsupervised Learning Network)
    * GAN (Generative Adversarial Network)
    * 編碼器與解碼器 (Encoder and Decoder): [Dimensionality Reduction in Tensorflow](data/EncoderDecoder_Tensorflow.html)
* 空間與圖影像
  * [資料擴增方法](data/ImageDataAugmentation.html)
  * 圖像分類 (Image Classification): 
    * AlexNet
    * VGGNet: [VGGNet in Keras](data/Keras_VGGNet_Tensorboard.html)
    * Inception V1, V2, V3
    * MobileNet
    * ResNet
    * SENet
  * 物件偵測 (Object Detection)
    * Faster R-CNN
    * Single Shot MultiBox Detector (SSD)
    * Mask R-CNN
    * You Only Look Once (YOLO): [darkflow](https://github.com/thtrieu/darkflow)
  * 圖像分割 (Image Segmentation)
    * Fully Convolutional Network (FCN)
    * U-Net
  * 物件追蹤
      * [Tracking with darkflow](https://github.com/bendidi/Tracking-with-darkflow)
  * 影像風格 (Artistic Style)
  * 光學字元識別 (Optical Character Recognition, OCR)
  * 立體影像 (3D)
  * 動作辨識 (Action Recognition)
  * Vedio Description
  * 圖像自動編輯與上色
  * 圖像資料集
    * [MS-COCO](http://cocodataset.org/#home)
    * [ImageNet](http://www.image-net.org/)
    * [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
    * [VisualQA](http://www.visualqa.org/)
    * [The Street View House Numbers](http://ufldl.stanford.edu/housenumbers/)
    * [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
    * [Multiple Object Tracking Benchmark](https://motchallenge.net/)
  * 影像資料集
    * [YT8M](https://research.google.com/youtube8m/) 
    * [UCF101](http://crcv.ucf.edu/data/UCF101.php)
    * [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#dataset)
    * [BDD100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
* 時間, 語言與音訊
  * 組成單元
    * 長短期記憶模型 (Long Short Term Memory Network, LSTM)
    * Gated Recurrent Unit (GRU)
    * Neural Tuning Machine (NTM)
    * Differentiable Neural Computer (DNC)
  * Word to Vector: [Word Embedding in Tensorflow](data/WordEmbedding_Tensorflow.html)
  * Independent Sequence: [Part of Speech in Tensorflow](data/seq2seq_PartOfSpeech.html)
  * Dependent Sequence: [RNN LSTM in Tensorflow](data/RNN_LSTM_Tensorflow.html)
  * 語言資料集
    * [IMDB Reviews](http://ai.stanford.edu/~amaas/data/sentiment/): [Introduce IMDB Dataset](data/IMDB_Dataset.html)
    * [Twenty Newsgroups](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)
    * [Sentiment140](http://help.sentiment140.com/for-students/)
    * [WordNet](https://wordnet.princeton.edu/)
    * [Yelp Reviews](https://www.yelp.com/dataset)
    * [The Wikipedia Corpus](http://nlp.cs.nyu.edu/wikipedia-data/)
    * [The Blog Authorship Corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm)
    * [Machine Translation of Various Languages](http://statmt.org/wmt18/index.html)
    * [Google News Dataset](https://code.google.com/archive/p/word2vec/)
    * [WMT'15](http://www.statmt.org/wmt15/translation-task.html)
    * Corpus: LSICC
  * 音訊資料集
    * [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
    * [Free Music Archive](https://github.com/mdeff/fma)
    * [Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html)
    * [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/)
    * [LibriSpeech](http://www.openslr.org/12/)
    * [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
  * 對話資料集
      * Persona-Chat
* Reinforcement Learning
* 示意推論
  * 圖像標注 (Image Caption)
  * Visual Question Answering
  * 文字推論資料集
      * [bAbI](https://research.fb.com/downloads/babi/)
* 自駕車 (Auto-Driving)
  * 辨識道路 
    * SegNet 
* 深度學習視覺化
    * [Tensorflow PlayGround](http://playground.tensorflow.org)
    * [TensorBoard](data/Tensorboard.html)

### 深度學習遭遇的問題

* 區域最佳解 (Local Optimal)
* 過度訓練或訓練不足
* 隱藏層深度及神經元數目
* 自動調整超參數 
* Gradient Vanishing

### 深度學習推論與應用

- Inference from Tensorflow Model
    - Classification Model
    - Object Detection Model
- Tensorflow Lite
    - [API](data/TensorflowLite_API.html)
    - [Command Line](data/TensorflowLite_CommandLine.html)
    - [In Respberry Pi ](data/TensorflowLite_RaspberryPi.html)
- Tensorflow XLA
- Tensorflow Debugger
- Distributed Tensorflow System
- Tensorflow with Kubernetes and Docker
- Tensorflow on Spark
- Tensorflow on android
- Tensorflow on iOS
- Tensorflow on Raspberry

### 學習流程管理

* [MLFlow](https://mlflow.org/)
* [FGLab](https://kaixhin.github.io/FGLab/)
* [Sacred](https://github.com/IDSIA/sacred)

### 其他框架的操作

* Onnx with Tensorflow: [Transform](data/Onnx_Tensorflow.html)

### Others Resource

* ~~CNTK (no more update)~~
  * Quick Start: [Python](data/CNTK_Quickstart_Python.html)
  * Vott-CNTK Flow: [Doc](data/vott_cntk_flow.html)