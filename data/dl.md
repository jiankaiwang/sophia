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

### MLP

### 神經網路

* 活化函式
	* Sigmoid
	* tanh
	* ReLu
	* Softplus
* 多層神經網路
* 輸出函式
	* 恆等式
	* Softmax
* 使用 MNIST 資料  

### 神經網路學習

* 損失函數
	* Mean Square Error
	* Cross Entropy Error
* 微分與偏微分
* 求梯度
	* 梯度下降 (Gradient Descent)
	* 梯度下降與學習

### 神經網路組合

* 使用正向傳播梯度下降法訓練的神經網路
* 計算權重的梯度
* 學習過程

### 批次學習

* 導入批次的學習

### 反向傳播法

* 多層架構
	* 正向傳播
	* 反向傳播
* 活化函式的正向與反向傳播
* 全連接層的正向與反向傳播
* 輸出函式的正向與反向傳播
* 梯度確認
* 反向傳播的學習
* 反向傳播的損失度

### 進階學習

* 最佳優化方法
	* Stochastic Gradient Descent (SGD)
	* Momentum
	* AdaGrad
	* Adam
	* nesterov
	* RMSprop

* 權重參數的預設值
	* Xavier method
	* Kaiming He method

* Batch Normalization
* Overfitting 與 Normalization
	* Weight Decay / Regularization
	* Dropout

* 超參數的設定
	* Validation data
	* 尋找最佳超參數的方法 

### 深度學習的類型與主流方法

* 基礎神經網路
	* 監督式學習網路 (Supervised Learning Network)
	    * MLP (Multi-Layer Preceptron)
	    * CNN (Convolutional Neural Network)
	    	* Convolutional Layer
	    	* Filter / Mask / Kernel
	    	* Padding
	    	* Stride
	    	* Batch
	    	* Pooling Layer 
	    	* im2col / col2im
	    * RNN (Recurrent Neural Network)
	* 非監督式學習網路 (Unsupervised Learning Network)
	    * GAN (Generative Adversarial Network)

* 空間與圖影像
	* 資料擴增方法
	* 圖像分類 (Image Classification)
		* LeNet 
		* AlexNet
		* VGG16, VGG19
		* GoogLeNet
		* Inception V1, V2, V3
		* MobileNet
		* ResNet
		* SENet
	* 物件偵測 (Object Detection)
		* R-CNN
		* Fast R-CNN
		* Faster R-CNN
		* Single Shot MultiBox Detector (SSD)
		* You Only Look Once (YOLO)
	* 圖像分割 (Image Segmentation)
		* Fully Convolutional Network (FCN)
		* U-Net
	* 影像風格 (Artistic Style)
	* 光學字元識別 (Optical Character Recognition, OCR)

* 時間與語音
	* 機器翻譯
		* 長短期記憶模型 (Long Short Term Memory Network)

* 示意推論
	* 編碼器與解碼器 (Encoder and Decoder)
		* seq2seq 
	* 圖像標注 (Image Caption)

* 自駕車 (Auto-Driving)
	* 辨識道路 
		* SegNet 

### 深度學習遭遇的問題

* 區域最佳解 (Local Optimal)，舉 SGD 為例
* 過度訓練或訓練不足
* 隱藏層數目及神經元數目




