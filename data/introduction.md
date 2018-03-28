### 機器學習的流程

* Deciding the Data Type: 決定訓練資料的型態，如透過圖像，語音等方式
* Collecting the Training Dataset: 蒐集訓練資料，需含有真實世界的特徵
* **Data Reading**: 讀入資料
* **Data Preprocessing**: 進行資料前處理
* Extracting the Learning Feature: 決定學習函數的輸入特徵及其表示法
* **Create a Model**: 決定學習函數及其對應的學習演算法使用的資料結構
* **Learning the Model Parameters**: 完成設計
* **Evaluating the Model**: 測試與預測
* Evoluting the Model: 持續性演化

### 連續資料離散化 (Discretization)

* 專家觀點
* 等分法
* S-Plus
    * modification of Sturges's rule that ensures nice break points between the bins
* Entropy Maximization Heuristic
* Bayesian Belief Network

### 資料降維/特徵萃取

* Principal Components Analysis (PCA): [R](data/Principal_Components_Analysis_R.html)
* Kernel Principal Components Analysis (KPCA)
* Factor Analysis: R
* 字典學習
* Nonparametric Weighted Feature Extraction (NWFE)
* Linear Discriminate Analysis (LDA): [R](data/Linear_Discriminate_Analysis_R.html)
* Quadratic Discriminate Analysis (QDA): [R](data/Quadratic_Discriminant_Analysis_R.html)
* Mixture of Generalized Gaussian (MGG)
* Mixture of Gaussian (MoG)

### 矩陣分解

* Singular Value Decomposition (SVD)
* QR Decomposition
    * Q is an orthogonal matrix 
    * R is an upper triangular matrix

### 最佳優化方法

* Stochastic Gradient Descent (SGD)
    * momentum_sgd
* Unconstrained and constrained minimization of multivariate scalar functions (minimize)  
    * BFGS
    * Nelder-Mead simplex
    * Newton Conjugate Gradient
    * COBYLA
    * SLSQP
* Global (brute-force) optimization routines 
    * basinhopping
    * differential_evolution
* Least-squares minimization (least_squares) and curve fitting (curve_fit) algorithms
* Scalar univariate functions minimizers (minimize_scalar) and root finders (newton)
* Multivariate equation system solvers (root) using a variety of algorithms
    * hybrid Powell
    * Levenberg-Marquardt
    * large-scale methods such as Newton-Krylov
* fsadagrad
* adam
* nesterov
* rmsprop

### 迴歸/預測

* Linear Regression: [R](data/Simple_Linear_Regression_R.html)
* Logistic Regression: [R](data/Logistic_Regression_R.html)
    * 二(多)元羅吉斯迴歸 (Logistic Regression Model): [R](data/Multinomial_Log-linear_Models_R.html)


### 機器學習的類型與主流方法

* 監督式學習, 分類演算法
    * 最近鄰居法 (KNN)
    * 類神經網路 (Neural Network)
    * 支援向量機 (SVM)
    * 隨機森林 (Random Forest)
    * 高斯混和模型 (Gaussian Mixture Model, GNN)
    * 高斯過程 (Gaussian Processes, GP)
    * 單純貝氏 (Naive Bayes)
    * 卡方自動交互檢視法決策樹 (CHAID for Decision Tree): [R](data/CHAID_R.html)
    * 放射狀基底函數網路 (Radial basis function, RBF)
  
* 非監督式學習, 分群演算法
    * 相鄰結合 (Neighbor Joining)
    * K-平均 (K-means)
    * 階層式分群法 (Hierarchical Clustering)
    * Fuzzy C-Means
    * Spectral Clustering
  
* 強化學習 (Reinforcement Learning)
    * Deep Q-Learning

* 遺傳演算法 (Genetic Algorithm, GA)

### 機器學習面臨的問題

* 高維度資料的處理
    * 樣本數與維度
    * Hughes Phenomenon
    * 皆為已知類別的解決方法: 
        * 降低維度
        * 正規化參數估計
    * 部分已知類別及部分未知類別的解決方法
        * Co-training
        * Reweighting
        * Common-component mixture with EM
        * Adaptive Classifier
* 半監督式學習
    * EM with Generative Mixture Model
    * Self-training
    * Co-training
    * TSVM
    * Mincut
    * Boltzmann Machine
    * Tree-based Bayes
* 最小誤差與計算成本問題
    * 達到誤差最小: 如最小平方法
    * 達到變異最小: 如貝式達到最小分類錯誤率
* 結構性的預測模式

### 深度學習的類型與主流方法

* 監督式學習網路 (Supervised Learning Network)
    * Deep Belief Network (DBN)
    * Multi-Layer Preceptron (MLP)
    * Feedforward Neural Network
    * Feedback Neural Network
    * CNN
    * RNN: CNTK
        * LSTM: CNTK
    * 稀疏編碼 (Sparse Encoding)
    * Back-Propagation Neural Network
    * Learning Vector Quantization
    * GAN
    
* 非監督式學習網路 (Unsupervised Learning Network)Each
    * 自我組機對應 (SOM)
    * 可適應式反響理論 (Adaptive Resonance Theory)

* 聯想式學習網路 (Associate Learning Network)
    * 霍普菲爾網路 (Hopfield Neural Network)
    * 雙向聯想記憶網路 (Bi-Directional Associative Memory)
    
* 最適化應用網路 (Optimization Application Network)
    * 霍普菲爾-坦克網路 (Hopfield-Tank Neural Network)
    * 退火網路 (Annealed Neural Network)
    
### 深度學習遭遇的問題

* 區域最佳解 (Local Optimal)，舉 SGD 為例
* 過度訓練或訓練不足
* 隱藏層數目及神經元數目
* 無法收斂




