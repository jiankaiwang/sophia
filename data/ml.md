### 機器學習的流程

* Data Reading: 讀入資料
* Data Preprocessing: 進行資料前處理
* **Extracting the Learning Feature**: 決定學習函數的輸入特徵及其表示法
* Create a Model: 決定學習函數及其對應的學習演算法使用的資料結構
* Learning the Model Parameters: 完成設計
* Evaluating the Model: 測試與預測

### 連續資料離散化 (Discretization)

* 專家觀點
* 等分法
* Entropy Maximization Heuristic
* Bayesian Belief Network

### 資料降維/特徵萃取

* Principal Components Analysis (PCA): [Rscript](data/Principal_Components_Analysis_R.html)
* Kernel Principal Components Analysis (KPCA)
* Factor Analysis:  [Rscript](data/Factor_Analysis_R.html)
* Multidimensional Scaling (MDS)
* Nonparametric Weighted Feature Extraction (NWFE)
* Linear Discriminate Analysis (LDA): [Rscript](data/Linear_Discriminate_Analysis_R.html)
* Quadratic Discriminate Analysis (QDA): [Rscript](data/Quadratic_Discriminant_Analysis_R.html)
* Mixture of Generalized Gaussian (MGG)
* Mixture of Gaussian (MoG)
* Lsomap
* Laplacian eigenmaps
* Locally-linear Embeddings
* t-SNE

### 矩陣分解

* Singular Value Decomposition (SVD)
* QR Decomposition

### 最佳優化方法

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

### 迴歸/預測

* Linear Regression: [Rscript](data/Simple_Linear_Regression_R.html)
* Logistic Regression: [Rscript](data/Logistic_Regression_R.html)
    * 二(多)元羅吉斯迴歸 (Logistic Regression Model): [Rscript](data/Multinomial_Log-linear_Models_R.html)
* Cox Regression: [Rscript](data/Cox_Regression_R.html)


### 機器學習的類型與主流方法

* 監督式學習, 分類演算法
    * 最近鄰居法 (KNN): [Rscript](data/K_Nearest_Neighbor_R.html)
    * 類神經網路 (Neural Network): [Rscript](data/Neural_Network_R.html)
    * 支援向量機 (SVM): [Rscript](data/Support_Vector_Machine_R.html)
    * 隨機森林 (Random Forest): [Rscript](data/Random_Forest_R.html)
    * 高斯混和模型 (Gaussian Mixture Model, GNN)
    * 高斯過程 (Gaussian Processes, GP)
    * 單純貝氏 (Naive Bayes) 與貝式網路 (Bayes Network): [Rscript](data/Bayes_R.html)
    * 卡方自動交互檢視法決策樹 (CHAID for Decision Tree): [Rscript](data/CHAID_R.html)
* 非監督式學習, 分群演算法
    * K-平均 (K-means): [Rscript](data/K_Means_R.html)
    * X-平均 (X-means)
    * 階層分群法 (Hierarchical Clustering): [Rscript](data/Hierarchical_Clustering_R.html)
        * 近鄰結合法: [Doc](data/Neighbor_Joining.html)
    * Fuzzy C-Means: [Rscript](data/Fuzzy_C-Means_R.html)
    * Spectral Clustering
    * Self-Organizing Map (SOM): [Rscript](data/Self-Organizing_Map_R.html)
* 遺傳演算法 (Genetic Algorithm, GA)

### 機器學習資料集

* Hard Drive Disk: https://www.backblaze.com/b2/hard-drive-test-data.html

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