## Project Title

Building a CNN model to identify defective jarlids.

## Project Overview

In modern manufacturing, automation plays a critical role in enhancing efficiency, consistency,
and quality control. One key area where automation can add significant value is in defect 
detection during the production process. This project focuses on automating the inspection of 
jarlids by developing a Convolutional Neural Network (CNN) model to classify defective versus 
non-defective jarlids.

Manual inspection is not only time-consuming but also prone to human error and 
inconsistency—making it unsuitable for large-scale operations. By leveraging deep learning and 
computer vision techniques, we aim to streamline the quality control process, reduce inspection
time, and increase the accuracy of defect identification. 

Our CNN-based model is trained on labeled image data to learn distinguishing features between 
acceptable and defective jarlids, enabling real-time and scalable quality assessment on the production line.

This solution supports the broader goal of smart manufacturing and Industry 4.0 by reducing 
reliance on manual labor and ensuring higher product reliability.

## Data Sources

We used a well-documented dataset from Kaggle, ensuring it was suitable for image classification 
task.

## Methodology

To improve the model's performance and generalizability, we began by augmenting the dataset 
using a variety of image transformation techniques. The goal was to artificially increase the 
number and diversity of training examples to help the model better learn relevant features. 
Our approach involved the following steps:

### 1. Data Augmentation

We applied several augmentation techniques to increase the size and variability of the dataset:

- **Rotations**: Images were rotated at angles of 90°, +35°, and -35° to simulate different 
                 orientations of jarlids.
- **Flipping**: Horizontal flip was used to expose the model to mirrored versions of the jarlids.
- **Lighting Adjustments**: Brightening and darkening were applied to help the model learn 
                            under varied lighting conditions.

### 2. Image Manipulation for Feature Enhancement

To help the CNN learn meaningful and discriminative features, we incorporated preprocessing 
techniques that emphasize key visual patterns:

- **X-ray Style Transformation**: Inverted color channels to highlight internal contrasts.
- **Edge Detection (Sobel Filter)**: Applied to capture contours and edges critical for detecting defects.
- **Image Sharpening**: Enhanced edges and textures, making defect features more distinguishable.

These preprocessing and augmentation steps were essential in creating a robust and 
comprehensive training set, ensuring the model learns not just from the limited original 
dataset, but also from a wide range of visually altered variations.


## Model Architecture

Our final model is a deep Convolutional Neural Network (CNN) built using TensorFlow/Keras, 
specifically tailored for binary classification of defective vs. non-defective jarlids.

### Input

- **Image Size**: 128×128
- **Channels**: 1 (grayscale)
- **Input Shape**: `(128, 128, 1)`

### Convolutional Layers

We use three blocks of convolutional layers to progressively extract complex features:

1. **Conv2D**: 32 filters, 5×5 kernel, `'same'` padding, `he_normal` initializer  
   → **LeakyReLU** activation (slope = 0.1)  
   → **MaxPooling2D**: 2×2 pool size

2. **Conv2D**: 64 filters, 5×5 kernel, `'same'` padding, `he_normal` initializer  
   → **LeakyReLU** activation  
   → **MaxPooling2D**

3. **Conv2D**: 128 filters, 5×5 kernel, `'same'` padding, `he_normal` initializer  
   → **LeakyReLU** activation  
   → **MaxPooling2D**

### Dense Layers

- **Flatten Layer**: Converts 2D feature maps into a 1D feature vector
- **Dense Layer**: 256 units, `ReLU` activation, `he_normal` initializer
- **Dropout Layer**: 30% dropout rate to reduce overfitting
- **Output Layer**: 2 units (logits), no activation (since we use `from_logits=True` in the loss function)

### Compilation & Training

- **Loss Function**: `SparseCategoricalCrossentropy(from_logits=True)`
- **Optimizer**: `Adam` with learning rate = 0.001
- **Metrics**: Accuracy
- **Batch Size**: 256
- **Epochs**: Up to 50, with **Early Stopping** (patience = 3) based on validation accuracy

This architecture was selected after extensive experimentation with hyperparameters and 
activation functions. It effectively balances model complexity with generalization capability, 
achieving high accuracy on both validation and unseen test data.

## Model Evaluation

This section presents the evaluation of our final CNN model's performance across training, 
validation, and test datasets using standard classification metrics.

---

### Accuracy & Loss Trends Over Epochs

- **Training Accuracy** improved consistently from ~51% to ~96% over 32 epochs.
- **Validation Accuracy** plateaued around **90%** after epoch 20, indicating model convergence.
- **Training Loss** decreased steadily to ~0.05, while **Validation Loss** plateaued around ~0.35.
- A growing gap between training and validation metrics suggests **moderate overfitting** starting around epoch 15–20.

> **Insight:**  
> The model learns quickly and generalizes well early on. Although mild overfitting is present, performance remains strong and stable.

---

### Validation Set Performance

| Class   | Correct | Incorrect | Total | Accuracy (%) |
|---------|---------|-----------|--------|---------------|
| Intact  | 2017    | 149       | 2166   | **93.1%**     |
| Damaged | 1769    | 249       | 2018   | **87.7%**     |
| **Overall** | —   | —         | 4184   | **90.4%**     |

---

### Test Set Performance

| Class   | Correct | Incorrect | Total | Accuracy (%) |
|---------|---------|-----------|--------|---------------|
| Intact  | 2035    | 132       | 2167   | **93.9%**     |
| Damaged | 1759    | 259       | 2018   | **87.2%**     |
| **Overall** | —   | —         | 4185   | **90.5%**     |

- Performance remains consistent with the validation set.

---

### ROC Curve & AUC Analysis

- **AUC Score**: **0.97** — indicates **excellent discriminative power**
- The ROC curve is tightly aligned with the **top-left corner**, reflecting:
  - High **True Positive Rate**
  - Low **False Positive Rate**
  - Strong separation between classes

> **Implication:**  
> The model is **robust across different classification thresholds** and maintains high generalization on unseen data.

---

### Summary

- **High performance** across both validation and test sets (≈ 90% accuracy)
- **AUC of 0.97** demonstrates strong capability to distinguish defective vs. intact jarlids
- **Slight overfitting** after ~20 epochs, but not performance-breaking

---

This evaluation confirms the CNN model is reliable, generalizes well, and is well-suited for 
deployment in an automated defect detection pipeline.

## Future Improvements

To further enhance the model's performance and robustness, several future directions can be 
explored:

1. **Leverage Pre-trained Models**  
   Utilizing transfer learning with well-established architectures such as **VGG16**, **ResNet50**, or **EfficientNet**. 
   These models, pre-trained on large-scale datasets like ImageNet, can provide powerful feature extraction capabilities 
   with minimal training data.

2. **Deeper Architectures**  
   Experimenting with **deeper CNNs** that include additional convolutional and pooling layers. This may help the model 
   capture more abstract, high-level features crucial for identifying subtle defects.

3. **Mixed Activation Functions**  
   Explore using different **activation functions** (e.g., ReLU, Leaky ReLU, PReLU, Swish) across various layers instead 
   of a single activation type. This can introduce non-linear diversity and improve model expressiveness.

4. **Inception-based Architectures**  
   **Inception modules**, which combine multiple filter sizes within the same layer to capture features at different 
   scales. This may be especially effective for capturing both fine and coarse structural details in jarlid images.

5. **Data Augmentation Strategies**  
   Continuing expanding the dataset with **advanced augmentation** (e.g., elastic deformation, random occlusion, 
   noise injection) to simulate real-world variability and further improve generalization.

---

By implementing these improvements, future iterations of the model can become more accurate, resilient, and suitable for 
real-time deployment in industrial settings.


## License
This project is licensed under the MIT License – see the LICENSE file for details.

## Contributions
If you'd like to contribute to this project, please fork the repository and submit a pull request. Ensure that your 
contributions follow the coding style and include tests where applicable.

## Credits

This project would not have been possible without the contributions and tools made available by the broader machine 
learning community.

- **Yann LeCun**, **Geoffrey Hinton**, and **Alex Krizhevsky** – for pioneering work on **Convolutional Neural Networks (CNNs)**, 

- **TensorFlow** – for providing a powerful, open-source framework that enabled efficient model development and training.

- **Kaggle** – for hosting a high-quality dataset that served as the basis for training and evaluating the model.

- **scikit-learn team** – especially **David Cournapeau** and **Matthieu Brucher**, for developing essential tools for 
model evaluation and metrics.

- **Springboard** – for providing a structured platform and curriculum to guide applied machine learning projects.

- **Karthik Ramesh** – my Springboard mentor, whose continuous support, guidance, and constructive feedback greatly 
enhanced the quality and direction of this project.
