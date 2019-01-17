# Sophia



![Build Status](https://cicd.ddns.net/buildStatus/icon?job=sophia)



The Sophia project is the tutorial for machine intelligence (MI), artificial intelligence (AI), basic mathematics, basic statistics. The whole document and implement are mainly based on Python and R, and several topics are implemented via other programming languages, i.e. C++, Javascript, Nodejs, Swift and Golang, etc. In this project, we introduce machine learning (ML) / deep learning (DL) models mainly based on sci-kit learn, Keras, and Tensorflow, and some topics might rely on other related libraries. We try to provide everyone with more applicable tutorials for MI and AI. 

Please enjoy it.



## Content

*   [Machine Learning](machine_learning/)

    *   [Dimensionality reduction](machine_learning/dimensionality_reduction/)
        -   Principal Components Analysis (PCA) : [Rscript](machine_learning/dimensionality_reduction/Principal_Components_Analysis_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/pca)
        -   Factor Analysis (FA) : [Rscript](machine_learning/dimensionality_reduction/Principal_Components_Analysis_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/fa)
        -   Multidimensional Scaling (MDS)
        -   Linear Discriminate Analysis (LDA) : [Rscript](machine_learning/dimensionality_reduction/Linear_Discriminate_Analysis_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/lda)
        -   Quadratic Discriminate Analysis (QDA) : [Rscript](machine_learning/dimensionality_reduction/Quadratic_Discriminant_Analysis_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/qda)
        -   t-SNE

    *   [Regression](machine_learning/regression/)
        -   Linear Regression : [Rscript](machine_learning/regression/Simple_Linear_Regression_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/slr)
        -   Logistic Regression : [Rscript](machine_learning/regression/Logistic_Regression_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/lr)
        -   Multinomial Logistic Regression Model : [Rscript](machine_learning/regression/Multinomial_Log-linear_Models_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/mllm)
        -   Cox Regression : [Rscript](machine_learning/regression/Cox_Regression_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/cr)

    *   [Supervised Learning](machine_learning/supervised_learning/)

        -   K-nearest neighbor (KNN) : [Rscript](machine_learning/supervised_learning/K_Nearest_Neighbor_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/knn)
        -   Neighbor Joining : [MD](machine_learning/supervised_learning/Neighbor_Joining.md), [webpage](https://sophia.ddns.net/machine_learning/supervised_learning/Neighbor_Joining.html)
        -   Neural Network : [Rscript](machine_learning/supervised_learning/Neural_Network_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/nn)
        -   Supported Vector Machine (SVM) : [Rscript](machine_learning/supervised_learning/Support_Vector_Machine_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/svm)
        -   Random Forest : [Rscript](machine_learning/supervised_learning/Random_Forest_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/rf)
        -   Gaussian Mixture Model (GMM)
        -   Naive Bayes and Bayes Network : [Rscript](machine_learning/supervised_learning/Bayes_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/bayes)
        -   CHAID for Decision Tree : [Rscript](machine_learning/supervised_learning/CHAID_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/chaid)

    *   [Un-Supervised Learning](machine_learning/unsupervised_learning/)

        -   K-means : [Rscript](machine_learning/unsupervised_learning/K_Means_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/kmeans)
        -   Hierarchical Clustering : [Rscript](machine_learning/unsupervised_learning/Hierarchical_Clustering_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/hclust)
        -   Fuzzy C-Means : [Rscript](machine_learning/unsupervised_learning/Fuzzy_C-Means_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/fcm)
        -   Self-Organizing Map (SOM) : [Rscript](machine_learning/unsupervised_learning/Self-Organizing_Map_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/som)

     *   Sci-Kit Learn: A famous machine learning library
*   [Deep Learning](deep_learning/)
    *   [Basis](deep_learning/basis)
        *   Components

            -   Activation Functions: Sigmoid, tanh, ReLu, Softplus
            -   Multi-Layer Network
            -   Output Functions: Softmax
        *   Learning

            -   Loss Functions
            -   Differential and Partial differential
            -   Gradient Descent and Learning
            -   Forward Propagation and Gradient Descent
            -   Weights' Gradients
            -   Learning Process
            -   Batch Learning
        *   Back Propagation

            -   Multi-layer architecture: Forward vs. Backward Propagation
            -   Learning via Backward Propagation
            -   Loss values on Backward Propagation
        *   Advanced Learning

            -   Optimization: SGD, Momentum, AdaGrad, Adam, nesterov, RMSprop
            -   Local / Glocal Minima with Optimizers: [ipynb](deep_learning/basis/BasicLearning_Tensorflow.ipynb), [webpage](https://sophia.ddns.net/deep_learning/basis/BasicLearning_Tensorflow.html)
            -   Initializing Weights: Xavier method, Kaiming He method
            -   Batch Normalization
            -   Overfitting / Normalization: Weight Decay / Regularization, Dropout
            -   Hyper-parameters: finding the best hyper-parameters
    *   [Space / Image](deep_learning/space_image/) : This topic focuses on deep learning theory in 2D image and 2D, 3D space.
        *   Data Preprocessing
            *   Data Augmentation : [ipynb](deep_learning/space_image/ImageDataAugmentation.ipynb) , [webpage](https://sophia.ddns.net/deep_learning/space_image/ImageDataAugmentation.html)
        *   Convolution Neural Network
            *   CNN Basis and Batch Normalization: [ipynb](deep_learning/space_image/BasicCNN_Tensorflow.ipynb), [webpage](https://sophia.ddns.net/deep_learning/space_image/BasicCNN_Tensorflow.html)
        *   Image Classification
            *   VGGNet in Keras : [ipynb](deep_learning/space_image/Keras_VGGNet_Tensorboard.ipynb), [webpage](https://sophia.ddns.net/deep_learning/space_image/Keras_VGGNet_Tensorboard.html)
            *   Tensorflow Image Classification Retrain (using NasNet as the example) : [ipynb](deep_learning/space_image/Keras_VGGNet_Tensorboard.ipynb), [webpage](https://sophia.ddns.net/deep_learning/space_image/Keras_VGGNet_Tensorboard.html)
            *   Inference from a frozen model (using tfhub' pretrained model) : [ipynb](deep_learning/space_image/Classification_Inference_from_PB_Tensorflow.ipynb), [webpage](https://sophia.ddns.net/deep_learning/space_image/Classification_Inference_from_PB_Tensorflow.html)
            *   Human Face : https://github.com/nyoki-mtl/keras-facenet
        *   Object Detection
            *   Faster R-CNN
            *   SSD + Classification Algorithms
            *   Yolo : [Author Github](https://github.com/pjreddie/darknet), [Github (Tensorflow)](https://github.com/thtrieu/darkflow)
            *   Tensorflow Object Detection API : [Github](https://github.com/jiankaiwang/TF_ObjectDetection_Flow)
            *   Inference from a frozen model (retrained via the above API) : [ipynb](deep_learning/space_image/object_detection_demo.ipynb), [webpage](https://sophia.ddns.net/deep_learning/space_image/object_detection_demo.html)
        *   Image Segmentation
            *   Mask R-CNN
    *   [Time / Series / Sequence](deep_learning/time_series_sequences) : This topic focuses on deep learning theory in data whose structure is like sequence, or time series.
        *   Data Preprocessing
            *   IMDB Dataset : [ipynb](deep_learning/time_series_sequences/IMDB_Dataset.ipynb), [webpage](https://sophia.ddns.net/deep_learning/time_series_sequences/IMDB_Dataset.html)
        *   Word Embedding
            *   Skip-Gram : [ipynb](deep_learning/time_series_sequences/WordEmbedding_Tensorflow.ipynb), [webpage](https://sophia.ddns.net/deep_learning/time_series_sequences/WordEmbedding_Tensorflow.html)
        *   Recurrent Neural Network
            -   Seq2Seq for Part of Speech (Vanilla RNN) : [ipynb](deep_learning/time_series_sequences/seq2seq_PartOfSpeech.ipynb), [webpage](https://sophia.ddns.net/deep_learning/time_series_sequences/seq2seq_PartOfSpeech.html)
            -   Static RNN vs. Dynamic RN : [ipynb](deep_learning/time_series_sequences/Simple_Dynamic_Seq2Seq.ipynb), [webpage](https://sophia.ddns.net/deep_learning/time_series_sequences/Simple_Dynamic_Seq2Seq.html)
        *   Bidirectional Dynamic RNN + LSTM / GRU cell + Attention Mechanism 
            *   Tensorflow 1.0.0 API : [ipynb](deep_learning/time_series_sequences/Seq2Seq_Bidirectional_Attention_tf-1.0.0.ipynb), [webpage](https://sophia.ddns.net/deep_learning/time_series_sequences/Seq2Seq_Bidirectional_Attention_tf-1.0.0.html)
            *   Tensorflow 1.11.0 API : [ipynb](deep_learning/time_series_sequences/Bidirectional_Dynamic_Seq2Seq.ipynb), [webpage](https://sophia.ddns.net/deep_learning/time_series_sequences/Bidirectional_Dynamic_Seq2Seq.html)
        *   Differentiable Neural Computer (DNC) : [Github](https://github.com/jiankaiwang/dnc-py3)
        *   Emotion Analyzing (IMDB Dataset) : [ipynb](deep_learning/time_series_sequences/RNN_LSTM_Tensorflow.ipynb), [webpage](https://sophia.ddns.net/deep_learning/time_series_sequences/RNN_LSTM_Tensorflow.html)
    *   Time and Space : This topic focuses on deep learning theory to the problems which are combining both space and time issues.

        *   Video
            *   Object Tracking
            *   Action Recognition
    *   [Encoder / Decoder](deep_learning/encoder_decoder/)
        *   AutoEncoder vs. PCA : [ipynb](deep_learning/encoder_decoder/EncoderDecoder_Tensorflow.ipynb), [webpage](https://sophia.ddns.net/deep_learning/encoder_decoder/EncoderDecoder_Tensorflow.html)
    *   [Generative Adversarial Network, GAN](deep_learning/gan)
        * Vanilla GAN : [ipynb](deep_learning/gan/SimpleGAN_Keras.ipynb), [webpage](https://sophia.ddns.net/deep_learning/gan/SimpleGAN_Keras.html)
        * Multiple Generator GAN : [ipynb](deep_learning/gan/MultiGenerator_GAN_Keras.ipynb), [webpage](https://sophia.ddns.net/deep_learning/gan/MultiGenerator_GAN_Keras.html)
        * DCGAN (Deep Convolutional GAN)
            -   https://github.com/jacobgil/keras-dcgan
        * CycleGAN (image-to-image translation)
            -   https://github.com/xhujoy/CycleGAN-tensorflow
        * SSGAN (Semi-supervised Learning GAN)
            -   https://github.com/gitlimlab/SSGAN-Tensorflow
    *   [Reinforcement Learning](deep_learning/reinforcement_learning)
        *   Basis Concept : [ipynb](deep_learning/reinforcement_learning/ReinforcementLearning_BasisConcept.ipynb), [webpage](https://sophia.ddns.net/deep_learning/reinforcement_learning/ReinforcementLearning_BasisConcept.html)
        *   Introduction to OpenAI Gym: [pyscript](deep_learning/reinforcement_learning/OpenAIGym_Introduction.py)
        *   Policy Learning (PoleCart) : [Concept.ipynb](deep_learning/reinforcement_learning/PoleCart_PolicyLearning.ipynb), [pyscript](deep_learning/reinforcement_learning/PoleCart_PolicyLearning.py)
        *   DQN (Value Learning, Breakout) : [Concept.ipynb](deep_learning/reinforcement_learning/Q_Learning_Concept.ipynb), [pyscript](deep_learning/reinforcement_learning/dqn.py)
    *   [Keras](deep_learning/keras) : A high level and python-based deep learning library which supports using Tensorflow, CNTK, etc. as its backend engine.
        *   Basis workflow using CNN : [ipynb](deep_learning/keras/Keras_Quickstart.ipynb), [webapge](https://sophia.ddns.net/deep_learning/keras/Keras_Quickstart.html)
        *   Editing Network and Convert to TFLite : [ipynb](deep_learning/keras/NetworkEditing_TFLite_Keras.ipynb), [webpage](https://sophia.ddns.net/deep_learning/keras/NetworkEditing_TFLite_Keras.html)
    *   [Tensorflow](deep_learning/tensorflow)
        * Basis

            -   Quick Start and Environment : [ipynb](deep_learning/tensorflow/Tensorflow_Quickstart_Python.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/Tensorflow_Quickstart_Python.html)
            -   Tensorflow Basis using MLP network : [ipynb](deep_learning/tensorflow/Basic_Tensorflow.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/Basic_Tensorflow.html)
            -   Tips in Tensorflow : [ipynb](deep_learning/tensorflow/Tips_Tensorflow.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/Tips_Tensorflow.html)
            -   **TF-Slim** Library
            -   Tensorflow Debugger
            -   Network Editing : [ipynb](deep_learning/tensorflow/NetworkEditing_Tensorflow.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/NetworkEditing_Tensorflow.html)
            -   Write out a frozen model
        * Data Preprocessing
            * Write data as a tfrecord format
            * Read data from a tfrecord format
            * tf.Data API
        * Tensorboard : a visualization tool designed to monitor deep learning task on Tensorflow

            * Basis : [ipynb](deep_learning/tensorflow/Tensorboard.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/Tensorboard.html)
            * CNN Example : [ipynb](deep_learning/tensorflow/CNN_Tensorboard.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/CNN_Tensorboard.html)
        * Cloud Computing

            -   Distributed Tensorflow System
            -   Tensorflow with Kubernetes and Docker
        * Mobile

            -   Tensorflow on android
            -   Tensorflow on iOS
        * Tensorflow Lite

            -   API introduction and convertion : [ipynb](deep_learning/tensorflow/TensorflowLite_API.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/TensorflowLite_API.html)
            -   Advanced convertion and network editing : [ipynb](deep_learning/tensorflow/TensorflowLite_CommandLine.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/TensorflowLite_CommandLine.html)
        * Retraining / tfhub
            * inference : [ipynb](deep_learning/tensorflow/tfhub_quickstart.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/tfhub_quickstart.html)
            * image retraining example based on pretrained models : [ipynb](deep_learning/tensorflow/tfhub_image_classification.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/tfhub_image_classification.html)
        * AdaNet (AutoML)
        * Tensorflow.js
        * Tensorflow Serving
        * Convertion with other frameworks
            * Convertion with Onnx : [ipynb](deep_learning/tensorflow/Onnx_Tensorflow.ipynb), [webapge](https://sophia.ddns.net/deep_learning/tensorflow/Onnx_Tensorflow.html)
        * Hyperparameter tuning
        * Distributed
            * MNIST_Replica : [pyscript](deep_learning/tensorflow/mnist_replica.py)
            * MNIST_CNN_Replica : [pyscript](deep_learning/tensorflow/mnist_cnn_replica.py)
*   [UI Tools](ui_tools/)
    *   [MLflow](ui_tools/mlflow)
        *   MLFlow quick tutorial: [ipynb](ui_tools/mlflow/mlflow_basis.ipynb), [webpage](https://sophia.ddns.net/ui_tools/mlflow/mlflow_basis.html)
*   [Mathematics / Statistics](mathematics_statistics/)

    * Linear algebra
    * Calculus
    * Probability 
    * [Statistics](mathematics_statistics/statistics/)
      * Basis : [MD](mathematics_statistics/statistics/basis.md)
      * Quantile Normalization : [Rscript](mathematics_statistics/statistics/Quantile_Normalization_R.rmd)
      * Pearson's Correlation : [Rscript](mathematics_statistics/statistics/Correlation_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/correlation)
      * Spearman's Correlation : [Rscript](mathematics_statistics/statistics/Correlation_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/correlation)
      * Logistic Regression : [Rscript](machine_learning/regression/Logistic_Regression_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/lr)
      * Multinomial/Ordinal Logistic Regression : [Rscript](machine_learning/regression/Multinomial_Log-linear_Models_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/mllm)
      * Fisher's exact test : [Rscript](mathematics_statistics/statistics/Fisher_Exact_Test_R.rmd)
      * Hypergeometeric test : [Rscript](mathematics_statistics/statistics/Hypergeometeric_test_R.rmd)















