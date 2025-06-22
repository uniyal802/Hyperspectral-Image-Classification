# Hyperspectral-Image-Classification
Hyperspectral Image Classification (HSI), focuses on using the rich spectral
details captured by hyperspectral sensors to accurately identify different types of land cover. Unlike
multispectral images, which often fall short when trying to distinguish between materials with
similar spectral characteristics, hyperspectral data provides information across hundreds of narrow
bands, making it possible to detect subtle differences. However, this also increases the complexity
of the data, requiring powerful techniques to process it effectively. To address this, the project
applies deep learning methods such as Convolutional Neural Networks (CNNs) and Bidirectional
Long Short-Term Memory (Bi-LSTM) networks. These models are designed to manage large, high-
dimensional datasets and improve the overall classification performance. The expected outcome is
a robust classification model that can be applied in areas like environmental monitoring, agriculture,
and urban development.

# Problem Statement
Hyperspectral imaging (HSI) captures detailed spectral data across hundreds of narrow bands, making it useful for land classification, agriculture, and urban planning.
However, its high dimensionality and limited labeled data present challenges. 
This project proposes a deep learning model combining wavelet decomposition and a CNN-BiLSTM hybrid. 
Haar wavelet transform extracts multiscale frequency features, which are processed using 3D CNNs and BiLSTM to capture spatial, spectral, and sequential patterns. 
The model enhances classification accuracy and robustness, outperforming traditional methods on standard HSI datasets.
This approach merges signal processing and deep learning for better hyperspectral image analysis.

# Project Objectives:
1. Improve Classification Accuracy: By applying advanced deep learning methods, with 3D and
   2D CNNs along with Bi-LSTMs, this project aims to significantly improve classification
   accuracy across different land cover types in hyperspectral datasets.
   
2. Handle High Dimensionality: The project seeks to address the challenges posed by the high-
   dimensional nature of hyperspectral data, using techniques like PCA for dimensionality
   reduction while retaining key spectral features.

3.  Create a Robust Model: The primary goal is to develop a model that can generalize well across
    multiple hyperspectral datasets (Indian Pines, Salinas, Botswana, and Pavia Centre) to ensure
    that it performs well in real-world applications.

4.  Enhance Computational Efficiency: By incorporating Bi-LSTMs which consist of two LSTM
    layers, in which one processes the input sequence in forward direction and the other processes
    it in reverse direction. These layers provide the output from both directions and are then
    combined to provide more comprehensive context at each time step and reducing the number of
    input features through PCA, the project aims to maintain a balance between classification
    accuracy and computational efficiency.

# Identified Challenges
1. High Dimensionality: Hyperspectral images typically have hundreds of spectral bands,
   leading to a large volume of data, which can be computationally expensive and prone to
   overfitting.


2. Spatial-Spectral Information Fusion: While spatial and spectral information are critical
   for accurate classification, integrating these features in a way that maximizes classification
   accuracy remains a complex task.


3. Feature Redundancy: Many spectral bands exhibit strong correlations, which can lead to I
   nefficient models that fail to extract meaningful features.

# Solution Approach
Our workflow for hyperspectral image classification is designed in such a way that the overall
procedure is implemented in an optimized manner that enables maximum utilization of deep
learning approaches for feature extraction and classification. The key steps in the workflow include
Raw Data Processing, Preprocessing, Feature Extraction, Model Design, Model Evaluation,
Classified Results, and Accuracy Assessment.

1. HSI Raw Data:
  The raw data comprises hyperspectral images captured using sensors that measure electromagnetic
  radiation across many spectral bands. Hyperspectral data is often high-dimensional, with each pixel
  containing information from multiple spectral bands. The Indian Pines dataset along with Salinas
  dataset, provide a range of hyperspectral data useful for classification tasks. These images contain
  vital information about land cover, vegetation types, and other geospatial features, but are often too
  large and complex for direct processing.

2. Data Preprocessing:
  In this phase, the raw HSI data is prepared for classification. This step includes several critical
  processes:
  Noise Removal: Hyperspectral data can contain noise from sensor limitations, which may
  reduce classification accuracy. Preprocessing methods like spectral filtering and denoising
  are applied to clean the data.

  Dimensionality Reduction: Given the high dimensionality of hyperspectral data,
  techniques like Principal Component Analysis (PCA) are used to reduce redundancy while
  retaining essential spectral information. This is crucial to improving model efficiency and
  preventing overfitting.

  Data Normalization: To standardize the data for uniform processing, pixel values are
  often normalized. This step ensures that variations in illumination and environmental
  factors do not influence the classification model.

3. Feature Extraction:
  Feature extraction plays a crucial role in hyperspectral image classification, as it involves the
  identification of relevant spatial and spectral features that distinguish between different classes. 3D
  CNNs, 2D CNNs and Bidirectional Long Short-Term Memory (Bi - LSTMs) are commonly used
  for feature extraction, where the former captures both spatial and spectral patterns across the multi-
  band images This is particularly useful in hyperspectral datasets with fine spatial-spectral
  interrelationships.

4. Model Design:
  For classification, various models are designed depending on the complexity of the data and the
  application:
  3D Convolutional Neural Networks (CNNs) are employed to process the spectral-spatial
  information of HSI data simultaneously.
  2D Convolutional Neural Networks (CNNs) are employed to extract the features of
  hyperspectral images.
  Hybrid models that combine 3D CNNs, 2D CNNs and Bi-LSTMs can further enhance
  classification by leveraging both spatial and spectral hierarchies.

5. Model Evaluation:
  After training the model, it is evaluated based on several metrics:
  Confusion Matrix Used to determine how well the model can distinguish between
  different classes.
  Accuracy, Precision, and Recall are metrics that assess the overall performance and class-
  wise performance of the model.
  Cross-validation can be employed to ensure that the model generalizes well across
  different subsets of the data.

6. Classified Results:
  Once the model has been trained and evaluated, it is applied to classify new, unseen hyperspectral
  images. The output is a land cover classification map that assigns each pixel to a specific class, such
  as vegetation, water bodies, urban areas, etc. The results are visualized, allowing users to interpret
  the model’s predictions and identify regions of interest.

7. Accuracy Assessment:
  Finally, the model's accuracy is thoroughly assessed using the results obtained from the classification
  process:
  Quantitative Accuracy Measures (e.g., overall accuracy, kappa coefficient) are used to
  assess how well the model performed against the ground truth data.
  Visual Inspection: Users may also perform visual inspections of the classified maps to
  validate the model’s accuracy by comparing them with known reference da
