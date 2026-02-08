# English-Fluency-Classification
A deep learning-based English fluency classification system using MFCC and spectrograms, featuring a comparative study of MLP, CNN, RNN, ResNet, and Vision Transformer (ViT).
1. Project Overview
This project presents an automated system for classifying English speaking fluency using various Deep Learning architectures. The goal is to provide students with immediate feedback on their oral English proficiency, thereby reducing the manual assessment workload for educators.The system processes raw audio by extracting MFCC (Mel-frequency Cepstral Coefficients) features and converting them into spectrogram representations. A comparative study was conducted across multiple neural network architectures:Standard Architectures: MLP, CNN, and RNN.Advanced Architectures: ResNet and Vision Transformer (ViT) (originally designed for computer vision but adapted here for audio classification).Key Finding: Most models achieved over 80% accuracy, while the ResNet and ViT models demonstrated superior performance, reaching a testing accuracy of nearly 90%.
2. Author's Note & Research Background
I completed this research project while attending high school in China. Thus, many of the original technical documents, notes, and datasets were recorded in Chinese.If you wish to review the detailed research process, experimental logic, and full analysis in English, please refer to the following file in this repository:👉 AI-based English Fluency Classification (English Version).pdf
This PDF is a translated version of my original paper and reflects my complete experimental methodology and thought process.
4. Technical Stack
Programming Language: Python
Audio Processing: Librosa, Pydub
Deep Learning Frameworks: TensorFlow / Keras, PyTorch
Feature Engineering: 25-dimensional MFCC extraction, 10-second non-overlapping audio segmentation.
5. Experimental Results
The models were trained and tested on a dataset of 447 audio segments (70% training, 30% testing).Model ArchitectureAccuracy (Approx.)MLP / CNN / RNN80%+ResNet~90%Vision Transformer (ViT)~90%
