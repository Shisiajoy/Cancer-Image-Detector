## MAMMOGRAM ANOMALY DETECTOR 🩺🥼

### Overview

This project aims to develop an anomaly detection system for mammogram images, where the anomalies are indicative of cancerous lesions. By leveraging autoencoders, the aim is to enhance early detection capabilities in breast cancer screening, providing valuable support for medical professionals.

### Abstract

Early detection of breast cancer significantly increases the chances of successful treatment. However, traditional image classification approaches may not always effective. This project focuses on employing autoencoders to detect anomalies, where cancerous images are classified as anomalies.

Typically, a radiologist might have to manually review thousands of mammogram images. With our autoencoder, the aim is to create a tool that will help the medical community in the pre-screening phase, to minimize human error and effort in recognizing possible positive cases.


#### Autoencoders

Autoencoders are a type of neural network used for unsupervised learning tasks. They are particularly effective for anomaly detection because they can learn to reconstruct input data. When presented with an image that is similar to the training data (i.e. a normal mammogram), the autoencoder will produce a low reconstruction error. Conversely, when it encounters an anomalous image (such as one depicting cancer), the reconstruction error will be significantly higher.

![image](https://github.com/user-attachments/assets/a6e1faec-199c-4016-8267-b9f8708bbbf0)


### Methodology

#### Dataset Description:

Trainning was done using the RSNA Screening Mammography Breast Cancer detection Dataset. The Dataset consists of 4 images for each person with a whole size of 54707 images. In this Dataset there are 1158 of Positive samples and 53549 negative samples.



![Screenshot 2024-12-06 125751](https://github.com/user-attachments/assets/cb04c155-3e78-4bc7-b8b6-ce8b1b8d96df)

#### Dataset Preprocessing:

Re-organize: Reorganized into images with and without cancer.

Resize: The image resolution is lowered from 5000 * 5000 to 500 * 500.

Reformat: The mammograms in the original dataset are in dicom format. These are converted to PNG format.


#### Model Architecture:

The autoencoder was designed with an encoder-decoder structure. The encoder compresses the input images into a latent space representation, while the decoder attempts to reconstruct the original images from this representation.

#### Training:

The autoencoder is trained on exclusively non- cancer images, such that it perfectly reconstructs a non- cancer image with minimal loss. This way, when the model encounters a cancer positive image, it will give high loss on reconstruction which can be interpreted as a possible positive case.

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
