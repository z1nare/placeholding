-----

# GPU-Accelerated Speech Recognition Pipeline

This project uses a C++/CUDA application for high-performance audio preprocessing and a PyTorch script to train a CNN for speech classification. It classifies `.wav` files of the words "bird," "cat," and "dog." 🗣️

The core idea is to offload the computationally expensive Short-Time Fourier Transform (STFT) to the GPU using NVIDIA's cuFFT library, creating spectrograms that are then used as input for a neural network.

-----

## Pipeline Overview

### 1\. Preprocessing (C++/CUDA)

  * **Load Audio**: Reads raw `.wav` files from disk.
  * **Frame Signal**: Splits the audio signal into small, overlapping frames.
  * **GPU Transfer**: Moves frames to the GPU using pinned memory for high throughput.
  * **Batched FFT**: Executes a batched Fast Fourier Transform on all frames in a single call using `cufftExecR2C`.
  * **Magnitude Kernel**: A custom CUDA kernel calculates the log-magnitude of the FFT output in parallel to create the spectrogram.
  * **Save Features**: The resulting spectrograms are copied back to the CPU and saved as `.csv` files.

### 2\. Training (Python/PyTorch)

  * **Load Spectrograms**: A custom `Dataset` loads the `.csv` spectrograms.
  * **CNN Model**: A simple Convolutional Neural Network classifies the 2D spectrogram data.
  * **Training**: The model is trained on the GPU using a standard training loop with Cross-Entropy loss and an Adam optimizer.
  * **Save Model**: The final trained model weights are saved to `spectrogram_classifier.pth`.

-----

##  Dataset

The complete dataset, including the raw **`.wav` audio files** and the **pre-processed spectrogram `.csv` files**, is publicly available on Kaggle.

  * **Kaggle Dataset**: [SpeechRecognition with CUDA and Torch](https://www.kaggle.com/datasets/z1nare/speechrecognition-with-cuda-and-torch)

You can download the data from there and skip the C++/CUDA preprocessing step if you wish to proceed directly to training the model.

-----

##  Quickstart

### Prerequisites

  * NVIDIA GPU
  * NVIDIA CUDA Toolkit
  * g++ Compiler
  * Python 3.8+ & Pip

### Instructions

1.  **Get the Data**: Download the dataset from [Kaggle](https://www.kaggle.com/datasets/z1nare/speechrecognition-with-cuda-and-torch). Place the raw `data/` or pre-processed `processed_data/` directory in the project root.

2.  **(Optional) Preprocess from Scratch**: If you want to generate the spectrograms yourself, compile and run the CUDA code.

    ```bash
    nvcc -std=c++17 -o preprocess generate_spectrograms.cu -lcufft
    ./preprocess
    ```

3.  **Install & Train**: Install Python dependencies and run the training cell in Jupyter Notebook.

    ```bash
    pip install requirements.txt
    ```

-----

##  Project Structure

```
.
├── data/              # Raw .wav files (from Kaggle)
├── processed_data/    # Generated .csv spectrograms (from Kaggle or preprocess)
├── preprocess         # Compiled CUDA executable
├── preprocess.cpp
├── train.py
└── spectrogram_classifier.pth # Final trained model
```

-----

##  Results

The model achieves high accuracy on the validation set after 15 epochs.

**Final Validation Accuracy: 98.33%** ✨

```
Epoch 15/15 - Loss: 0.0647 - Train Acc: 98.15% - Val Acc: 98.33%
✅ Model saved to spectrogram_classifier.pth
```
