# Deep_fake_detection


## 📖 Project Overview
This project focuses on detecting deepfake videos using advanced deep learning techniques. Deepfakes represent a significant threat to digital media integrity, and this work explores two hybrid architectures to address this challenge: a **custom CNN+LSTM model** built from scratch and a **pre-trained ResNet50+LSTM model**. The goal is to accurately classify videos as **Real** or **Fake**.

---

## 🎯 Objectives
- Develop and compare two deep learning models for deepfake detection.
- Utilize spatial and temporal features from video frames using CNN and LSTM components.
- Achieve high accuracy and generalization on the Celeb-DF dataset.

---



## 🧠 Models Implemented

### 1. CNN + LSTM (From Scratch)
- A custom convolutional neural network for spatial feature extraction.
- Bidirectional LSTM layers to capture temporal dependencies.
- Trained end-to-end on the Celeb-DF dataset.

### 2. ResNet50 + LSTM (Transfer Learning)
- Uses a pre-trained ResNet50 model as a feature extractor.
- Features are passed to a bidirectional LSTM for sequence modeling.
- Fine-tuned on the target dataset for improved performance.

---

## 📊 Dataset
We used the **Celeb-DF (v2)** dataset [[Li et al., CVPR 2020]](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.html), which includes:
- 590 real videos
- 5,639 deepfake videos
- Over 2 million video frames
- Diverse in gender, age, ethnicity, and lighting conditions

  ### 🔖 Citation
If you use this dataset, please cite:

@inproceedings{li2020celebdf,
  title={Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
  author={Li, Yuezun and Yang, Xin and Sun, Pu and Qi, Honggang and Lyu, Siwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3207--3216},
  year={2020}
}


---

## ⚙️ Methodology

### 🧩 Data Preparation Pipeline

#### 1. Frame Extraction
- Extracted **20 frames per video** using systematic sampling  
- Implemented robust handling for corrupted or incomplete videos  
- Converted frames to **RGB format** for consist

#### 2. Preprocessing
- Resized all frames to **64×64 pixels** to reduce computational complexity  
- Normalized pixel values to **[0, 1]** range by dividing by 255  
- Applied **data shuffling** to prevent order-based bias  

#### 3. Dataset Splitting
- **Training set:** 80% of videos  
- **Test set:** 20% of videos  
- **Balanced sampling** to address class imbalance (590 real vs 5,639 fake videos)  

---

## 🏗️ Model Architectures

### 🔹 CNN + LSTM (From Scratch)

#### Feature Extraction (CNN)
- **Input:** 224×224×3 RGB images  
- Convolutional layers with (5×5) filters and ReLU activation  
- **Batch normalization** after each convolution  
- **Dilated convolutions** (rate=2) for expanded receptive field  
- **MaxPooling2D** for spatial downsampling  

#### Temporal Modeling (LSTM)
- **Bidirectional LSTM layers** for sequence processing  
- First layer returns full sequences for enriched context  
- Second layer returns final condensed vector  
- 512-unit dense layer with ReLU activation  

#### Training Configuration
- **Optimizer:** Adam (learning rate: 0.0001)  
- **Loss:** Categorical Cross-Entropy  
- **Batch size:** 32, **Epochs:** 20  
- **Callbacks:** ReduceLROnPlateau, EarlyStopping  

---

### 🔬 ResNet50 + LSTM (Transfer Learning)

#### Feature Extraction
- **Pre-trained ResNet50** on ImageNet as backbone  
- Frozen initial layers, fine-tuned last 50 layers  
- Output features: 7×7×2048 (49 spatial positions × 2048 features)  

#### Sequence Processing
- Feature maps reshaped to (49, 2048) sequences  
- **Bidirectional LSTM** with 128 units per direction  
- **Batch normalization** for training stability  

#### Classification Head
- Dense layers (1024, 512 units) with ReLU  
- **Dropout (0.5)** for regularization  
- **Softmax** output layer for binary classification  

#### Training Configuration
- **Optimizer:** Adam with adaptive learning rate  
- **Loss:** Categorical Cross-Entropy  
- **Metrics:** Accuracy  
- **Callbacks:** ReduceLROnPlateau, EarlyStopping with best weights restoration  

---

## 🧠 Training Strategy

### Regularization Techniques
- **Dropout:** Applied in dense layers to prevent overfitting  
- **Batch Normalization:** For stable training and faster convergence  
- **Early Stopping:** Stop training when validation loss plateaus  
- **Learning Rate Scheduling:** Reduce LR when validation performance stalls  

### Data Handling
- One-hot encoding for class labels  
- Real videos labeled as **0**, Fake videos as **1**  
- Frame sequences treated as **temporal data** for LSTM processing  

---

## 📈 Results

### Quantitative Performance

| Model             | Test Accuracy | Test Loss | Frames per Sequence |
|-------------------|---------------|------------|----------------------|
| **CNN + LSTM**    | 78%           | 0.30       | 20                   |
| **ResNet50 + LSTM** | 93%         | 0.18       | 20                   |

---

### Performance Analysis

#### CNN + LSTM Model
- Training Accuracy: ~85%  
- Validation Accuracy: ~80%  
- Moderate generalization with some overfitting tendencies  
- Stable learning curve with gradual convergence  

#### ResNet50 + LSTM Model
- Training Accuracy: ~98%  
- Validation Accuracy: ~90%  
- Excellent feature extraction capabilities  
- Minor overfitting managed by regularization  

---

### Key Findings
- **Transfer Learning Superiority:** ResNet50’s pre-trained features significantly boosted performance  
- **Temporal Modeling:** LSTM layers effectively captured sequential patterns in video frames  
- **Architecture Impact:** Hybrid CNN–LSTM outperformed single-modality approaches  
- **Computational Efficiency:** ResNet50 provided better features with similar computational cost  

---

## 🖥️ Web Application

We developed an intuitive **web interface using Streamlit** with the following features:

### Interface Components
- **Model Selection:** Choose between CNN+LSTM and ResNet50+LSTM models  
- **File Upload:** Support for video file input (.mp4 format)  
- **Real-time Analysis:** Frame extraction and processing visualization  
- **Results Display:** Confidence scores and classification output  

### Application Flow
1. User selects preferred detection model  
2. Uploads video file for analysis  
3. System extracts and displays frames  
4. Model processes frames and provides classification  
5. Results show confidence levels and final verdict  

### Features
- Responsive design for different screen sizes  
- Progress indicators during processing  
- Side-by-side comparison of models  
- Export capabilities for analysis results  

---

## 🚧 Challenges & Solutions

| Challenge | Impact | Solution Implemented |
|------------|---------|----------------------|
| **GPU Resource Limitations** | Slow training and limited experimentation | Utilized Kaggle GPU resources for training |
| **Class Imbalance** | Model bias toward majority class (fake videos) | Random sampling and balanced batch creation |
| **Variable Video Quality** | Inconsistent feature extraction | Standardized frame resolution (64×64) and normalization |
| **Computational Complexity** | Long training times for video data | Fixed frame sampling (20 frames/video) and optimized preprocessing |
| **Overfitting** | Poor generalization on unseen data | Implemented dropout, early stopping, and learning rate scheduling |

---

## 🔮 Future Work

### Model Improvements
- Incorporate **3D CNN architectures** for spatiotemporal features  
- Experiment with **Vision Transformers (ViT)** for global context understanding  
- Implement **attention mechanisms** in LSTM layers  
- Explore **self-supervised learning** for feature extraction  

### Technical Enhancements
- Real-time detection capabilities  
- Browser-based deployment without GPU requirements  
- Ensemble methods combining multiple architectures  
- Adaptive frame selection based on video content  

### Dataset Expansion
- Include more diverse deepfake generation techniques  
- Incorporate **audio analysis** for multimodal detection  
- Add challenging scenarios like **low-quality and compressed videos**

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+  
- TensorFlow 2.0+  
- OpenCV  
- Streamlit  
- NumPy, Pandas  

### Installation

# Clone repository
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection

# Install dependencies
pip install -r requirements.txt

### Installation

# Train models
python train_cnn_lstm.py
python train_resnet_lstm.py

# Run web application
streamlit run app.py
---
# ✅ Conclusion
The **ResNet50+LSTM** model demonstrated superior performance in deepfake detection, achieving **93% accuracy** with excellent generalization capabilities. This hybrid approach successfully combines the powerful feature extraction of pre-trained CNNs with the temporal modeling strength of LSTMs.

Key contributions of this work include:

Comprehensive comparison of from-scratch vs transfer learning approaches

Effective handling of temporal dependencies in video data

Practical web application for real-world deployment

Robust preprocessing pipeline for video data

This project highlights the potential of hybrid deep learning architectures in addressing the growing challenge of deepfake media and contributes to the development of reliable digital media verification tools.

---
# 👥 Contributors
- **Bouhnas Chaymae**

- **EL Alami Nihad**

- **EL Gamani Ahlam**

- **Supervised by**: Prof. Belcaid Anas
- **Academic Year**: 2024-2025
- **Institution**: École Nationale des Sciences Appliquées de Tétouan
---
# 📜 License
This project is for academic and research purposes. Please cite the original dataset and authors if used in research or publications.

---
# 🙏 Acknowledgments

- Celeb-DF dataset creators for providing high-quality benchmark data

- TensorFlow and Keras communities for excellent deep learning tools

- Kaggle for providing computational resources

- ENSA Tétouan for academic support and guidance


