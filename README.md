# Music Genre Recognition via CRNN and Improved ResNet-BiLSTM-Attention Architecture

This repository contains the implementation and experimental results for the term project of **MYZ307E - Machine Learning for Electrical and Electronics Engineering**.

The project focuses on **Music Genre Recognition (MGR)** using the GTZAN dataset. We first reproduced and adapted a legacy Keras-based CRNN architecture in PyTorch, then improved the model using a residual CNN front-end, BiLSTM layers, and temporal attention pooling.

## Team Members

- Ahmet Selim Pala
- Muhammed Ebrar Torun
- Mesut Anlak
- İsmail Emin Turhan

## Reference Work

This project builds upon the following paper:

K. Choi, G. Fazekas, M. Sandler, and K. Cho,  
"Convolutional Recurrent Neural Networks for Music Classification,"  
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017.

Original GitHub repository:

https://github.com/keunwoochoi/music-auto_tagging-keras

## Dataset

The experiments use the **GTZAN Music Genre Dataset**, consisting of 10 music genres:

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

After cleaning and preprocessing, 929 valid audio tracks were used. Each track was represented as 40 mel-spectrogram segments of size `128 × 130`, resulting in 37,160 segment-level samples.

The dataset was split at the song level:

- Training files: 743
- Validation files: 186
- Training segments: 29,720
- Validation segments: 7,440

The dataset itself is not included in this repository due to size limitations.

## Repository Structure

```text
music-genre-recognition-crnn/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── baseline/
│   │   ├── crnn_baseline_pytorch.py
│   │   └── train_crnn_baseline.py
│   │
│   └── improved/
│       ├── model_improved.py
│       ├── train_improved.py
│       └── train_improved_COLAB.ipynb
│
├── results/
│   ├── baseline_report.txt
│   ├── full_model_report.txt
│   ├── no_attention_report.txt
│   ├── resnet_only_report.txt
│   ├── training_history.json
│   ├── full_model_history.json
│   ├── no_attention_history.json
│   ├── resnet_only_history.json
│   ├── training_curves.png
│   ├── improved_training_curves.png
│   ├── confusion_matrix.png
│   ├── improved_confusion_matrix.png
│   └── ablation_study_comparison.png
│
└── docs/
