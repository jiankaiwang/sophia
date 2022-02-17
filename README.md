# Sophia



![Build Status](https://cicd.ddns.net/buildStatus/icon?job=sophia)



The Sophia project is the tutorial for `machine intelligence` (MI), `artificial intelligence` (AI), `basic mathematics`, `basic statistics`. The whole document and implement are mainly based on Python and R, and several topics are implemented via other programming languages, i.e. C++, Javascript, Nodejs, Swift and Golang, etc. In this project, we introduce machine learning (ML) / deep learning (DL) models mainly based on `sci-kit learn`, `Keras`, and `Tensorflow`, and some topics might rely on other related libraries. We try to provide everyone with more applicable tutorials for MI and AI. 

Please enjoy it.



## Content

### [Machine Learning](machine_learning/)

#### [Dimensionality reduction](machine_learning/dimensionality_reduction/)
* Principal Components Analysis (PCA): [Rscript](machine_learning/dimensionality_reduction/Principal_Components_Analysis_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/pca)
* Factor Analysis (FA): [Rscript](machine_learning/dimensionality_reduction/Principal_Components_Analysis_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/fa)
* Multidimensional Scaling (MDS)
* Linear Discriminate Analysis (LDA): [Rscript](machine_learning/dimensionality_reduction/Linear_Discriminate_Analysis_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/lda)
* Quadratic Discriminate Analysis (QDA): [Rscript](machine_learning/dimensionality_reduction/Quadratic_Discriminant_Analysis_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/qda)
* Singular Value Decomposition (SVD): [notebook](machine_learning/dimensionality_reduction/Singular_Value_Decomposition.ipynb)
* t-SNE

#### [Regression / Classification](machine_learning/classification_regression/)
* Linear Regression: [Rscript](machine_learning/classification_regression/Simple_Linear_Regression_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/slr)
* Logistic Regression: [Rscript](machine_learning/classification_regression/Logistic_Regression_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/lr)
* Multinomial Logistic Regression Model: [Rscript](machine_learning/classification_regression/Multinomial_Log-linear_Models_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/mllm)
* Cox Regression: [Rscript](machine_learning/classification_regression/Cox_Regression_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/cr)
* Regression via TF2Keras: [notebook](machine_learning/classification_regression/TF2_Regression.ipynb)

#### [Supervised Learning](machine_learning/supervised_learning/)

* K-nearest neighbor (KNN): [Rscript](machine_learning/supervised_learning/K_Nearest_Neighbor_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/knn)
* Neighbor Joining: [Markdown](machine_learning/supervised_learning/Neighbor_Joining.md), [webpage](https://sophia.ddns.net/machine_learning/supervised_learning/Neighbor_Joining.html)
* Neural Network: [Rscript](machine_learning/supervised_learning/Neural_Network_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/nn)
* Supported Vector Machine (SVM): [Rscript](machine_learning/supervised_learning/Support_Vector_Machine_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/svm)
* Random Forest: [Rscript](machine_learning/supervised_learning/Random_Forest_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/rf)
* Naive Bayes and Bayes Network: [Rscript](machine_learning/supervised_learning/Bayes_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/bayes)
* CHAID for Decision Tree: [Rscript](machine_learning/supervised_learning/CHAID_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/chaid)

#### [Un-Supervised Learning](machine_learning/unsupervised_learning/)

* K-means: [Rscript](machine_learning/unsupervised_learning/K_Means_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/kmeans)
* Hierarchical Clustering: [Rscript](machine_learning/unsupervised_learning/Hierarchical_Clustering_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/hclust)
* Fuzzy C-Means: [Rscript](machine_learning/unsupervised_learning/Fuzzy_C-Means_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/fcm)
* Self-Organizing Map (SOM): [Rscript](machine_learning/unsupervised_learning/Self-Organizing_Map_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/som)


### [Deep Learning](deep_learning/)

#### Theories and Mechanism

> This section is the tutorial of basic theories about deep learning. We discuss how to design a model and how to train it. Furthermore, we discuss lots of details about the mechanism of well-designed training and try to implement those from scratch. Please refer to the repository, [jiankaiwang/deep_learning_from_scratch](https://github.com/jiankaiwang/deep_learning_from_scratch), for more details. :link:

#### Structured Data

* Classify Structured Data using Feature Columns on `TF2.Keras`: [notebook](deep_learning/structured/Classify_with_FeatureColumns.ipynb)
* Classify Imbalanced Structured Data using `TF2Data` and `TF2.Keras`: [notebook](deep_learning/structured/Classification_on_Imbalanced_Data.ipynb)
* Time Series Forecasting on Structured Data using `TF2.Keras`: [notebook](deep_learning/structured/TF2Keras_Time_Series_Forecasting.ipynb)

#### Image / Space

* Dataset Processing
    * Cifar10 Dataset: [notebook](deep_learning/space_image/Cifar-10_Dataset.ipynb)
    * Data Augmentation: [notebook](deep_learning/space_image/ImageDataAugmentation.ipynb)
* Convolution Neural Network
    * CNN Basis and Batch Normalization: [notebook](deep_learning/space_image/BasicCNN_Tensorflow.ipynb)
* Image Classification
    * A CNN Example with Keras: [notebook](deep_learning/space_image/Keras_VGGNet_Tensorboard.ipynb)
    * Inference from a frozen model (using tfhub pretrained model): [notebook](deep_learning/space_image/Classification_Inference_from_PB_Tensorflow.ipynb)
    * image retraining example based on pretrained models: [notebook](deep_learning/space_image/tfhub_image_classification.ipynb)
    * A quick view of a `TF2.Keras` CNN model: [notebook](deep_learning/space_image/TF2Keras_Cifar10.ipynb)
    * A quick view of a CNN model with a Customized Layer via `TF2.Keras` and `TF2.Core` as well: [notebook](deep_learning/space_image/TF2Keras_MNIST_Classification.ipynb)
    * Image classification model with data augmentation and overfitting prevention using `TF2.keras`: [notebook](deep_learning/space_image/TF2Keras_Flow.ipynb)
    * Training with Transfer Learning using `TF2.Hub`: [notebook](deep_learning/space_image/TF2Hub_Transfer_Learning_Image_Classification.ipynb)
    * Transfer Learning with `Feature Extraction` and `Fine-Tuning` using `TF2.Keras`: [notebook](deep_learning/space_image/TF2Keras_Transfer_Learning_PreTrained_Model.ipynb)
* Object Detection
    > The introduction of object detection algorithms and the implementations from scratch were moved to the new repository, [jiankaiwang/object_detection_tutorial](https://github.com/jiankaiwang/object_detection_tutorial). :link:

    > We also introduce the Tensorflow Object Detection APIs for a shortcut of state-of-art model implementations. The content was moved to [jiankaiwang/TF_ObjectDetection_Flow](https://github.com/jiankaiwang/TF_ObjectDetection_Flow). :link:
* Image Segmentation
    * a Modified U-Net(a Semantic Network): [notebook](deep_learning/space_image/TF2Keras_Image_Semantic_Segmentation.ipynb)
    * Mask R-CNN
* Depth Estimator
    * Monocular Depth Estimation
#### Series / Sequence / Time
* Data Preprocessing
    * Dataset List: [markdown](deep_learning/time_series_sequences/dataset_list.md)
    * IMDB Dataset: [notebook](deep_learning/time_series_sequences/IMDB_Dataset.ipynb)
    * NLTK Introduction: [notebook](deep_learning/time_series_sequences/NLTK_Introduction.ipynb)
* Word Embedding
    * Word2Vec using Skip-Gram: [notebook](deep_learning/time_series_sequences/WordEmbedding_Tensorflow.ipynb)
    * Word embedding using `TF1.Hub`: [notebook](deep_learning/time_series_sequences/tfhub_word_embedding.ipynb)              
    * Train embeddings using `TF2Keras`: [notebook](deep_learning/time_series_sequences/TF2Keras_WordEmbeddings.ipynb)
* Text Classification
    * A quick view of implementing text classification via `Tensorflow 2` with `TF2.Keras` as well: [notebook](deep_learning/time_series_sequences/TF2Keras_IMDB_Classification.ipynb)
    * Advanced text classification with training embeddings using `TF2Keras`: [notebook](deep_learning/time_series_sequences/TF2Keras_TextClassification_RNN.ipynb)
* Recurrent Neural Network
    * Seq2Seq for Part of Speech (Vanilla RNN): [notebook](deep_learning/time_series_sequences/seq2seq_PartOfSpeech.ipynb)
    * Static RNN vs. Dynamic RN: [notebook](deep_learning/time_series_sequences/Simple_Dynamic_Seq2Seq.ipynb)
* Seq2Seq Model
    * Character-based Text Generation using `TF2Keras`: [notebook](deep_learning/time_series_sequences/TF2Keras_TextGeneration_RNN.ipynb)
    * Bidirectional Dynamic RNN + LSTM / GRU cell + Attention Mechanism using Tensorflow 1.0.0 [notebook](deep_learning/time_series_sequences/Seq2Seq_Bidirectional_Attention_tf-1.0.0.ipynb), or using Tensorflow 1.11.0 [notebook](deep_learning/time_series_sequences/Bidirectional_Dynamic_Seq2Seq.ipynb), or using TF2Keras [notebook](deep_learning/time_series_sequences/TF2Keras_NMT_Attention.ipynb)
    * Using the Transformer architecture on TF2Keras: [notebook](deep_learning/time_series_sequences/TF2Keras_Transformer_LanguageUnderstanding.ipynb)
* Differentiable Neural Computer (DNC)
    > Differentiable Neural Computer (DNC) is a kind of enhanced neural cell-like LSTM and GRU. It enhances the memory mechanism to memorize more long-term information. The more details please refer to the repository, [jiankaiwang/dnc-py3](https://github.com/jiankaiwang/dnc-py3). :link:
* Emotion Analyzing (IMDB Dataset): [notebook](deep_learning/time_series_sequences/RNN_LSTM_Tensorflow.ipynb)
#### Audio / Signal
* Dataset Processing:
    * Datasets: [markdown](deep_learning/audio_singal/datasets.md)
    * Data Formats: [markdown](deep_learning/audio_singal/data_formats.md)
    * Signal Processing: [notebook](deep_learning/audio_singal/Signal_Processing.ipynb)
    * Audio Signal Processing Example: [notebook](deep_learning/audio_singal/Audio_Processing.ipynb)
* Word-based
    > Word-level audio recognition is one of the most popular requirements in technical fields. Simple words like `Up`, `Down`, `Hey, Siri!` or `OK, Google!`, etc. are commonplace. The details about how to implement a word-level audio recognition were moved to a new repository `jiankaiwang/word_level_audio_recognition` (https://github.com/jiankaiwang/word_level_audio_recognition). :link:
* Phrase-based
* Sentence-based
* Environment Sound Classification: [notebook](deep_learning/audio_singal/Environment_Sound_Classification.ipynb)

#### Images with Text or Space with Time

* Image Captioning using Attention Mechanism using `TF2Keras`: [notebook](deep_learning/time_space/TF2Keras_ImageCaptioning_Attention.ipynb)

#### Encoder / Decoder

* AutoEncoder vs. PCA: [notebook](deep_learning/encoder_decoder/EncoderDecoder_Tensorflow.ipynb)
* Intro to Basic Autoencoders: [notebook](deep_learning/generative/Autoencoder.ipynb)

#### [Generative](deep_learning/generative)

* Vanilla GAN using Keras: [notebook](deep_learning/generative/Keras_SimpleGAN.ipynb)
* Multiple Generator GAN: [notebook](deep_learning/generative/MultiGenerator_GAN_Keras.ipynb)
* Introduction to Neural Style Transfer: [notebook on TF2Keras](deep_learning/generative/TF2Keras_NeuralStyleTransfer.ipynb)
* Introduction to DeepDream: [notebook on TF2Keras](deep_learning/generative/TF2Keras_DeepDream.ipynb)
* Introduction to DCGANs Using TF2Keras: [notebook](deep_learning/generative/TF2Keras_VanillaGAN_DCGAN.ipynb)
* Pix2Pix with Conditional GANs and PatchGANs: [notebook](deep_learning/generative/TF2Keras_Pix2Pix.ipynb)
* BigGAN From TF.Hub: [notebook](deep_learning/generative/BigGAN_TF_Hub_Demo.ipynb)
* Introduction to CycleGAN: [notebook](deep_learning/generative/TF2_CycleGAN.ipynb)
* Adversarial examples using FGSM: [notebook](deep_learning/generative/AdversarialExample_FGSM.ipynb)

#### Reinforcement Learning

* Basis Concept: [notebook](deep_learning/reinforcement_learning/ReinforcementLearning_BasisConcept.ipynb)
* Introduction to OpenAI Gym: [pyscript](deep_learning/reinforcement_learning/OpenAIGym_Introduction.py)
* Policy Learning (PoleCart): [notebook](deep_learning/reinforcement_learning/PoleCart_PolicyLearning.ipynb), [pyscript](deep_learning/reinforcement_learning/PoleCart_PolicyLearning.py)
* DQN (Value Learning, Breakout): [notebook](deep_learning/reinforcement_learning/Q_Learning_Concept.ipynb), [pyscript](deep_learning/reinforcement_learning/dqn.py)
    
### Frameworks

#### ml5.js

> `ML5.js` is the higher level API built on the top of `Tensorflow.js`.  This tutorial had been moved to a new repository [jiankaiwang/mljs](https://github.com/jiankaiwang/mljs).

#### Tensorflow.js

> We provide you with more details about lower level API to machine learning or deep learning tasks. This tutorial had been moved to a new repository [jiankaiwang/mljs](https://github.com/jiankaiwang/mljs). :link:                

#### [Tensorflow 2.x](frameworks/tensorflow2): 2019 ~ Now

* differences between version 1.x and 2.x
    * New Features to Datasets: [notebook](frameworks/tensorflow2/TF2_AddDatasets_Splits.ipynb)
    * New Features to the Eager Mode, TF functions, and Flow Controllers: [notebook](frameworks/tensorflow2/TF2_Eager_tfFunc.ipynb)
    * New Features to TF.Data, KerasHub, and Model Building: [notebook](frameworks/tensorflow2/TF2_tfdata_kerashub.ipynb)
    * Using `tf.function` and `py_function` in Advanced: [notebook](frameworks/tensorflow2/TF2_TFFunction_PYFunction.ipynb)
* Save and Load Models
    * Manipulation of SavedModel in TF2: [notebook](frameworks/tensorflow2/TF2_SavedModels.ipynb)
    * Save and Load Models in TF2Keras: [notebook](frameworks/tensorflow2/TF2Keras_Save_Load_Models.ipynb)
* The whole flow
    * Using `TF2.Keras` on the Structured Dataset: [notebook](frameworks/tensorflow2/TF2_StructuredFlow.ipynb)
    * Using `TF2.Keras` on Image Datasets: [notebook](frameworks/tensorflow2/TF2_Keras_Flow.ipynb)
    * Using `TF2.Keras` and `TF Core` in advanced: [notebook](frameworks/tensorflow2/TF2_Customization.ipynb)
* The Manipulation of Datasets on Tensorflow 2:
    * Processed CSV Files with `tf2.data` APIs: [notebook](frameworks/tensorflow2/TF2Data_CSV.ipynb)
    * Processed Numpy and Pandas Objects with `tf2.data` APIs: [notebook](frameworks/tensorflow2/TF2Data_Pandas_Numpy.ipynb)
    * Processed Image Datasets with `tf2.keras` and `tf2.data` APIs: [notebook](frameworks/tensorflow2/TF2Data_Image.ipynb)
    * Processed Text Datasets with `tf.data`: [notebook](frameworks/tensorflow2/TF2Data_Text.ipynb)
    * Processed Unicode Text Datasets with `tf.data`: [notebook](frameworks/tensorflow2/TF2Data_Unicode.ipynb)
    * Processed TF.Text with `tf.data`: [notebook](frameworks/tensorflow2/TF2Data_TFText.ipynb)
    * Processing `TFRecord` Files with `tf.train.Example`: [notebook](frameworks/tensorflow2/TF2Example_TFRecord.ipynb)
* Tensorflow Estimator(`tf.estimator`) on Tensorflow 2:
    * Premade Estimators: [notebook](frameworks/tensorflow2/TF2Estimator_Premade_Estimators.ipynb)
    * A Linear Model with TF.Estimator: [notebook](frameworks/tensorflow2/TF2Estimator_LinearModel.ipynb)
    * A Simple Boosted Trees Model with TF.Estimator: [notebook](frameworks/tensorflow2/TF2Estimator_SimpleBoostedTrees.ipynb)
    * Understanding Boosted Trees Model: [notebook](frameworks/tensorflow2/TF2Estimator_AdvancedBoostedTrees.ipynb)
    * Creating a TF.estimator Model From a TF.keras Model: [notebook](frameworks/tensorflow2/TF2Estimator_EstimatorFromKeras.ipynb)  

#### [Tensorflow 1.x](frameworks/tensorflow): 2015 ~ Now

* Basis Flow
    * Quick Start and Environment: [notebook](frameworks/tensorflow/Tensorflow_Quickstart_Python.ipynb)
    * Tensorflow Basis using MLP network: [notebook](frameworks/tensorflow/Basic_Tensorflow.ipynb)
    * Tips in Tensorflow: [notebook](frameworks/tensorflow/Tips_Tensorflow.ipynb)
    * Network Editing: [notebook](frameworks/tensorflow/NetworkEditing_Tensorflow.ipynb)
    * Write out a frozen model: [pyscript](frameworks/tensorflow/Generate_FrozenModel.py)
    * Manipulation of SavedModel in TF1: [notebook](frameworks/tensorflow/TF1_Save_Load_Models.ipynb)
* Data Manipulation
    * manipulating the `tfrecord` format: [notebook](frameworks/tensorflow/tf1_manipulating_tfrecords.ipynb)
    * `tf.Data` API
* Tensorflow Estimators (`tf.estimator`)
    * a numeric example of `Estimators` APIs: [notebook](frameworks/tensorflow/tf_estimator_example.ipynb)
    * an image example of `Estimator` APIs with serving example: [notebook](frameworks/tensorflow/TensorflowServing_ImageExample.ipynb)
* Tensorflow Keras (`tf.keras`)
    > The `TF1.Keras` APIs are similar to the official Keras APIs. However, there are some improvements in `TF1.keras`, e.g. for distributed training, etc. This section provides no more document or you can refer to `jiankaiwang/distributed_training` (https://github.com/jiankaiwang/distributed_training) :link: and an updated version `TF2.keras` ([Tensorflow 2.x](frameworks/tensorflow2)).             
* Tensorboard: a visualization tool designed to monitor progesses
    * Basis: [notebook](frameworks/tensorflow/Tensorboard.ipynb)
    * CNN Example: [notebook](frameworks/tensorflow/CNN_Tensorboard.ipynb)  
* Distributed architecture: 
    > Distributed architecture helps engineers or scientists to scale up the serving capability or to speed up the training process. Now it is easy to deploy a scalable serving architecture via `Tensorflow Serving`. Here we put an emphasis on how to speed up the training process. We have transferred the whole content to a new repository, `jiankaiwang/distributed_training` (https://github.com/jiankaiwang/distributed_training). :link:
* Tensorflow Lite
    > We moved this topic to a new repository, please refer to the repository [jiankaiwang/aiot](https://github.com/jiankaiwang/aiot) for more details. :link:
    * API introduction and convertion: [notebook](frameworks/tensorflow/TensorflowLite_API.ipynb)
    * Advanced convertion and network editing via shell commands: [notebook](frameworks/tensorflow/TensorflowLite_CommandLine.ipynb)
    * A flow from convertion to doing an inference: [notebook](frameworks/tensorflow/TFLite_FromFrozenModel_Inference.ipynb)
* Tensorflow Extended
    * Quick Start: [markdown](frameworks/tensorflow/Tensorflow_Extended.md)
    * Introduction to `ExampleGen`: [notebook](frameworks/tensorflow/TFX_ExampleGen.ipynb)
    * Introduction to `Tensorflow Data Validation` (`TFDV`): [notebook](frameworks/tensorflow/TFX_TFDV.ipynb)
    * Introduction to `Tensorflow Transform` (`tft`): [notebook](frameworks/tensorflow/TF_Transform.ipynb)
    * Introduction to `TFX Modeling`
    * Introduction to `Tensorflow Model Analysis (TFMA)`: [notebook](frameworks/tensorflow/TensorflowModelAnalysis_MNIST_MLPExample.ipynb)
    * Introduction to `Tensorflow Serving`: 
        * Article: Run via Docker: [markdown](frameworks/tensorflow/tfserving_docker.md)
        * Request a Restful API for the Numeric Prediction: [notebook](frameworks/tensorflow/TensorflowServing_NumericPrediction.ipynb)
        * request a Restful API for an Image Recognition: [notebook](frameworks/tensorflow/TensorflowServing_ImageExample.ipynb)
* AdaNet
    * An Example of the workflow: [notebook](frameworks/tensorflow/adanet_objective.ipynb)
    * Customized Model: [notebook](frameworks/tensorflow/customizing_adanet.ipynb)
* Convertion with other frameworks
    * Convertion with Onnx: [notebook](frameworks/tensorflow/Onnx_Tensorflow.ipynb)

#### [Keras](frameworks/keras): Here Keras in Sophia is the official Python library supporting multiple backends and also only supporting Tensorflow 1.x, not `tf.keras`.

* Basis workflow using CNN: [notebook](frameworks/keras/Keras_Quickstart.ipynb), [webapge](https://sophia.ddns.net/frameworks/keras/Keras_Quickstart.html)
* Editing Network and Convert to TFLite: [notebook](frameworks/keras/NetworkEditing_TFLite_Keras.ipynb), [webpage](https://sophia.ddns.net/frameworks/keras/NetworkEditing_TFLite_Keras.html)        

### [Other Tools](other_tools/)

#### [MLflow](other_tools/mlflow)

* MLFlow quick tutorial: [notebook](ui_tools/mlflow/mlflow_basis.ipynb), [webpage](https://sophia.ddns.net/ui_tools/mlflow/mlflow_basis.html)

#### Simple Cloud-AI (SCAI)

> Simple Cloud-AI (SCAI) is a simple Restful API server for AI image recognition. It provides a UI interface and two API services on the backend. Now it supports both image recognition and object detection tasks. You can also update your customized retained models no matter what it is coming from transfer learning or Tensorflow Object Detection API. In addition, you can integrate the APIs with your own service. You can refer to the repository, [jiankaiwang/scai](https://github.com/jiankaiwang/scai), for more details. :link:

#### AI x Iot (aiot)

> We integrate Raspberry Pi, Intel Movidius, and Jetson Tx2, etc. to an AIoT prototype architecture. We demonstrate how to use these integrated devices to do AI projects. Please refer to the repository [jiankaiwang/aiot](https://github.com/jiankaiwang/aiot) for more details. :link:

### [Mathematics / Statistics](mathematics_statistics/)

#### [Statistics](mathematics_statistics/statistics/)
* Basis: [Markdown](mathematics_statistics/statistics/README.md)
* Quantile Normalization: [Rscript](mathematics_statistics/statistics/Quantile_Normalization_R.rmd)
* Pearson's Correlation: [Rscript](mathematics_statistics/statistics/Correlation_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/correlation)
* Spearman's Correlation: [Rscript](mathematics_statistics/statistics/Correlation_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/correlation)
* Logistic Regression: [Rscript](machine_learning/regression/Logistic_Regression_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/lr)
* Multinomial/Ordinal Logistic Regression: [Rscript](machine_learning/regression/Multinomial_Log-linear_Models_R.rmd), [RPubs](https://rpubs.com/jiankaiwang/mllm)
* Fisher's exact test: [Rscript](mathematics_statistics/statistics/Fisher_Exact_Test_R.rmd)
* Hypergeometeric test: [Rscript](mathematics_statistics/statistics/Hypergeometeric_test_R.rmd)















