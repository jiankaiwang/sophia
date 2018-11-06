### 深度學習的流程

* Deciding the Data Type: 決定訓練資料的型態，如透過圖像，語音等方式
* Collecting the Training Dataset: 蒐集訓練資料，需含有真實世界的特徵
* **Data Reading**: 讀入資料
* **Data Preprocessing**: 進行資料前處理
* Extracting the Learning Feature: 決定學習函數的輸入特徵及其表示法
* **Create a Model**: 決定學習函數及其對應的學習演算法使用的資料結構
* **Learning the Model Parameters**: 完成設計
* **Evaluating the Model**: 測試與預測
* Evoluting the Model: 持續性演化

### 神經網路

* 活化函式: Sigmoid, tanh, ReLu, Softplus
* 多層神經網路
* 輸出函式: 恆等式, Softmax
* 使用 MNIST 資料  

### 神經網路學習

* 損失函數: Mean Square Error, Cross Entropy Error
* 微分與偏微分
* 求梯度: 梯度下降 (Gradient Descent), 梯度下降與學習

### 神經網路組合

* 使用正向傳播梯度下降法訓練的神經網路
* 計算權重的梯度
* 學習過程

### 批次學習

* 導入批次的學習

### 反向傳播法

* 多層架構: 正向傳播 vs. 反向傳播
* 活化函式的正向與反向傳播
* 全連接層的正向與反向傳播
* 輸出函式的正向與反向傳播
* 梯度確認
* 反向傳播的學習
* 反向傳播的損失度

### 進階學習

* 最佳優化方法: Stochastic Gradient Descent (SGD), Momentum, AdaGrad, Adam, nesterov, RMSprop

* 權重參數的預設值: Xavier method, Kaiming He method

* Batch Normalization
* Overfitting 與 Normalization: Weight Decay / Regularization, Dropout

* 超參數的設定: Validation data, 尋找最佳超參數的方法 

### 深度學習的類型與主流方法

* 基礎神經網路
  * 監督式學習網路 (Supervised Learning Network)
      * MLP (Multi-Layer Preceptron)
      * CNN (Convolutional Neural Network): 
        * Convolutional Layer, Filter / Mask / Kernel 
        * Padding, Stride
        * Pooling Layer
        * im2col / col2im
      * RNN (Recurrent Neural Network)
  * 非監督式學習網路 (Unsupervised Learning Network)
    * GAN (Generative Adversarial Network)
    * 編碼器與解碼器 (Encoder and Decoder)
* 空間與圖影像
  * [資料擴增方法](data/ImageDataAugmentation.html)
  * 圖像分類 (Image Classification)
    * AlexNet
    * VGG16, VGG19
    * Inception V1, V2, V3
    * MobileNet
    * ResNet
    * SENet
  * 物件偵測 (Object Detection)
    * Faster R-CNN
    * Single Shot MultiBox Detector (SSD)
    * Mask R-CNN
    * You Only Look Once (YOLO)
  * 圖像分割 (Image Segmentation)
    * Fully Convolutional Network (FCN)
    * U-Net
  * 物件追蹤
  * 影像風格 (Artistic Style)
  * 光學字元識別 (Optical Character Recognition, OCR)
  * 立體影像 (3D)
  * 動作辨識 (Action Recognition)
  * 圖像資料集
    * MS-COCO: [http://cocodataset.org/#home](http://cocodataset.org/#home)
    * ImageNet: [http://www.image-net.org/](http://www.image-net.org/)
    * Open Images Dataset: [https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
    * VisualQA: [http://www.visualqa.org/](http://www.visualqa.org/)
    * The Street View House Numbers: [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/)
    * CIFAR-10: [http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)
  * 影像資料集
    * YT8M: [https://research.google.com/youtube8m/](https://research.google.com/youtube8m/) 
    * UF101: [http://crcv.ucf.edu/data/UCF101.php](http://crcv.ucf.edu/data/UCF101.php)
    * MPII Human Pose Dataset: http://human-pose.mpi-inf.mpg.de/#dataset
    * BDD100k: https://bair.berkeley.edu/blog/2018/05/30/bdd/
* 時間, 語言與音訊
  * 機器翻譯
    * 長短期記憶模型 (Long Short Term Memory Network, LSTM)
    * Gated Recurrent Unit (GRU)
  * 語言資料集
    * IMDB Reviews: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/), [IMDB Dataset]()
    * Twenty Newsgroups: [https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)
    * Sentiment140: [http://help.sentiment140.com/for-students/](http://help.sentiment140.com/for-students/)
    * WordNet: [https://wordnet.princeton.edu/](https://wordnet.princeton.edu/)
    * Yelp Reviews: [https://www.yelp.com/dataset](https://www.yelp.com/dataset)
    * The Wikipedia Corpus: [http://nlp.cs.nyu.edu/wikipedia-data/](http://nlp.cs.nyu.edu/wikipedia-data/)
    * The Blog Authorship Corpus: [http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm)
    * Machine Translation of Various Languages: [http://statmt.org/wmt18/index.html](http://statmt.org/wmt18/index.html)
    * Google News Dataset: [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/)
    * WMT'15: http://www.statmt.org/wmt15/translation-task.html
  * 音訊資料集
    * Free Spoken Digit Dataset: [https://github.com/Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
    * Free Music Archive: [https://github.com/mdeff/fma](https://github.com/mdeff/fma)
    * Ballroom: [http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html)
    * Million Song Dataset: [https://labrosa.ee.columbia.edu/millionsong/](https://labrosa.ee.columbia.edu/millionsong/)
    * LibriSpeech: [http://www.openslr.org/12/](http://www.openslr.org/12/)
    * VoxCeleb: [http://www.robots.ox.ac.uk/~vgg/data/voxceleb/](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
  * 對話資料集
      * Persona-Chat
* 示意推論
  * 圖像標注 (Image Caption)
  * Visual Question Answering
  * 文字推論資料集
      * bAbI: https://research.fb.com/downloads/babi/
* 自駕車 (Auto-Driving)
  * 辨識道路 
  	* SegNet 

### 深度學習遭遇的問題

* 區域最佳解 (Local Optimal)
* 過度訓練或訓練不足
* 隱藏層深度及神經元數目
* 自動調整超參數 
* Gradient Vanishing




