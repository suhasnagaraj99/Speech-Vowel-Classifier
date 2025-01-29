# Speech-Vowel-Classifier

This repository demonstrates a **logistic regression** approach for **vowel classification** using **PyTorch**. The dataset is derived from the Hillenbrand corpus (1995), focusing on American English vowels produced in "hVd" contexts. We take mid-frame MFCC features as input to a PyTorch-based logistic regression model and evaluate classification accuracy on pairs of vowels (e.g., `ih` vs. `eh`).

---

## Repository Structure

- **lr_speech.py**
  Contains:
  - **`SpeechDataset`**: A PyTorch dataset wrapper for loading vowel data.
  - **`create_dataset`**: Function to load WAV files, compute 13 MFCCs, and extract the center frame for each vowel utterance (followed by z-scoring).
  - **`SimpleLogreg`**: A single-layer logistic regression model leveraging PyTorch’s `nn.Linear`.
  - **`step`**: A helper function that performs one iteration of forward pass, loss computation, backprop, and optimizer step.
  - **Main Section**: Argument parsing, data splitting, and training loop for multiple epochs.

---

## Prerequisites

- **Python 3.7+**
- **PyTorch** `>= 1.0`
- **Librosa** for MFCC feature extraction
- **NumPy**
- **scikit-learn** (for train/test splits)
- **SoundFile** (via `pip install soundfile`) for WAV reading
- **glob**, **argparse**, etc. (built-in libraries)

### Installing Required Packages (under a Python Virtual Environment)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset
The script expects a directory structure like:
```bash
Hillenbrand/
    men/
        08men_iy.wav
        ...
    women/
        08women_ah.wav
        ...
    children/
        ...
```
- Where each WAV file is named with its vowel label (e.g., iy.wav, ah.wav, etc.). The code extracts exactly two vowels (provided via the --vowels argument) from these subdirectories.

- Supported Vowels (subset from the dataset): ae, ah, aw, eh, ei, er, ih, iy, oa, oo, uh, uw

---

## Setup Instructions

- **Clone the Repository**
  ```bash
  git clone https://github.com/suhasnagaraj99/Speech-Vowel-Classifier.git
  cd Speech-Vowel-Classifier
  ```
- **Install Dependencies**
  ```bash
  pip install -r requirements.txt
  ```
- **Check Directory Structure**
  - Ensure you have the unzipped Hillenbrand/ directory (with subdirectories: men/, women/, children/) containing .wav files for the vowels you want to classify.

---

## Running the code

Example command:
```bash
python logreg_pytorch.py \
    --vowels ih,eh \
    --directory ./Hillenbrand \
    --num_mfccs 13 \
    --passes 5 \
    --batch 1 \
    --learnrate 0.1
```
  - `--vowels`: Comma-separated vowels to classify (e.g., ae,ah).
  - `--directory`: Path to the Hillenbrand data folder.
  - `--num_mfccs`: Number of MFCC coefficients (default: 13).
  - `--passes`: Number of epochs for training.
  - `--batch`: Batch size for SGD (default: 1).
  - `--learnrate`: Learning rate for the optimizer (default: 0.1).

- **Process**
  - The script scans the Hillenbrand data for the specified vowels.
  - It creates a dataset by reading all WAV files, computing MFCCs, extracting the midpoint frame, and z-scoring features.
  - It splits the data into train/test sets (default test size is 15%).
  - Initializes a logistic regression model via PyTorch (SimpleLogreg class).
  - Performs training across the specified number of epochs (passes).
  - Prints out loss and accuracy (train and test) every 20 steps.

---

## Model Details
- SpeechDataset:
  - Expects a NumPy array where the first column is the label (0 or 1) and the remaining columns are MFCC features.
  - Convert features and labels to PyTorch tensors.

- SimpleLogreg:
  - A single linear layer (nn.Linear(num_features, 1)) followed by a sigmoid activation in forward.
  - Uses BCELoss as the loss function.

- Optimization:
  - Uses stochastic gradient descent (torch.optim.SGD) or your optimizer of choice.
  - Learning rate default is 0.1 (configurable).
 
## Acknowledgement

- This project references the CMSC 723 course framework by Dr. Naomi Feldman and Dr. Jordan Boyd-Graber.
- Thanks to UMD for the continued support and resources.
- Special thanks to the open-source community (PyTorch, librosa, etc.) for tools enabling this project.

## Attribution
The original problems were provided as part of the course:
- Course: Natural Language Processing [CL1-HW](https://github.com/Pinafore/cl1-hw)
- License: CC-BY 4.0 [Link](https://creativecommons.org/licenses/by/4.0/legalcode)

