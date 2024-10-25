## MAMMOGRAM ANOMALY DETECTOR 🩺🥼

### Overview

This project aims to develop an anomaly detection system for mammogram images, where the anomalies are indicative of cancerous lesions. By leveraging autoencoders, the aim is to enhance early detection capabilities in breast cancer screening, providing valuable support for medical professionals.

### Problem Statement

Early detection of breast cancer significantly increases the chances of successful treatment. However, traditional image classification approaches may not always effective. This project focuses on employing autoencoders to detect anomalies, where cancerous images are classified as anomalies. The goal is to build a model that can reconstruct normal images while highlighting those that deviate significantly from the norm.

Typically, a radiologist might have to manually review thousands of mammogram images. However, with our autoencoder, the model analyzes every image first and flags only those that deviate from the normal pattern.

### Methodology

#### Autoencoders
Autoencoders are a type of neural network used for unsupervised learning tasks. They are particularly effective for anomaly detection because they can learn to reconstruct input data. When presented with an image that is similar to the training data (i.e., a normal mammogram), the autoencoder will produce a low reconstruction error. Conversely, when it encounters an anomalous image (such as one depicting cancer), the reconstruction error will be significantly higher. This property makes autoencoders ideal for our task.

#### Training Process
Data Preparation: We collected a dataset of mammogram images from RSNA mammogram dataset

Images were resized and normalized to enhance model performance.

#### Model Architecture:

The autoencoder was designed with an encoder-decoder structure. The encoder compresses the input images into a latent space representation, while the decoder attempts to reconstruct the original images from this representation.

#### Training:

The model was trained using mean squared error as the loss function, which measures the difference between the input images and their reconstructed outputs. The model learned to minimize this error through iterative updates to its weights.

#### Anomaly Detection:

After training, the reconstruction error was calculated for each image. A threshold was established based on the mean and standard deviation of these errors to identify anomalous images.

### Why Autoencoders?

Choosing autoencoders over traditional classification methods was driven by their ability of autoencoders to learn without explicit labels, making them particularly suitable for our scenario where we focus on detecting anomalies rather than classifying multiple categories.

### Results

The trained model was evaluated on a set of mammogram images, with a clear distinction between normal and cancerous images based on reconstruction loss. Results showed that images with high reconstruction errors were often those that depicted anomalies. This demonstrates the potential of autoencoders in aiding early cancer detection.

### Deployment
The application is deployed using Streamlit, allowing users to upload mammogram images and receive feedback on potential anomalies.

### Conclusion

This project highlights the importance of anomaly detection in medical imaging, specifically in identifying cancerous lesions in mammograms. By utilizing autoencoders, it provide a promising approach to enhance early detection efforts, potentially leading to better patient outcomes.

### 🦺Future Work

1. Further optimization of the autoencoder architecture and hyperparameters.

2. Exploration of ensemble methods to improve anomaly detection accuracy.

3. Integration of additional data sources for a more comprehensive analysis.
