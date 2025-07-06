```
# ██████═╗ ██████╗ ██╗   ██╗██╗████████╗
# ██╔══██╝ ██╔══██╗██║   ██║██║╚══██╔══╝
# ██████╗  ██████╔╝██║   ██║██║   ██║   
# ██╔══██╗ ██╔══██╗██║   ██║██║   ██║   
# ██████╔╝ ██║  ██║╚██████╔╝██║   ██║   
# ╚═════╝  ╚═╝  ╚═╝ ╚═════╝ ╚═╝   ╚═╝   
```
# Bioacoustic Recognition of Unhealthy and Intact Tones (BRUIT)

This framework is a modular framework to train models for bruit identification. Every parameter, from preprocessing to model architectures, are customizable from a local config file.  
Here is an example performance report from an example model trained using this framework.

```
Classification Report:
              precision    recall  f1-score   support

    artifact       0.96      0.88      0.92        89
     healthy       0.75      0.85      0.80        47
      murmur       0.84      0.87      0.85        61

    accuracy                           0.87       197
   macro avg       0.85      0.87      0.86       197
weighted avg       0.88      0.87      0.87       197
```

# Setup

BRUIT is easily setup using the shell script `setup.sh`. This script builds a python virtual environment and can be ran on Linux, Windows, and MAC.


### 1. Clone the Repository
First, clone the repository.

```
git clone https://github.com/NeuroSonicSoftware/Jacob-assignment.git
```

### 2. Setup the Environment
```
source setup.sh
```
This script will:

* Detect your platform (Linux, Windows via WSL or Git Bash, or macOS)

* Create a Python virtual environment in .venv/
 
* Activate the environment

* Install required packages the pyproject.toml (Mac GPU unavailable)

* Set up base configuration files

> [!NOTE]
> For GPU training, ensure you have a compatible NVIDIA driver and CUDA/cuDNN stack installed. The framework is tested with TensorFlow 2.15+.

> [!IMPORTANT]
> Note for Windows users: Native Windows is not officially supported. Use WSL2 (Windows Subsystem for Linux) or Git Bash for best results.

> [!WARNING]
> Note for macOS users: Due to hardware and driver limitations, TensorFlow and related tools will run in CPU-only mode. Training and inference will be slower, but fully functional.

### 3. Verify the Environment

Check to see if the BRUIT framework was properly setup by running the help command.
```
bruit -h
```
You should see the positional arguments if setup properly.
```
usage: bruit [-h] {preprocess,train} ...

BRUIT CLI - A command line interface for BRUIT image processing and analysis

positional arguments:
  {preprocess,train}
    preprocess        Run all preprocessing steps on input audio files (.wav)
    train             Train a model on preprocessed data

options:
  -h, --help          show this help message and exit
```
## Setting Parameters

The parameters for both the preprocessing and the training use yaml formatting and can be found in a config file `configs/parameters.yaml`. Currently, there are three sections to the parameters file, `preprocessing`, `model_training` and `plotting`. An example snippet from `model_training` is shown below.
```
####### Model Training ######
model_training:
  classifier: true
  training_feature: "mfcc"    # Current Trainable Features:  mfcc, mel, rms, zcr
  loss_function: "sparse_categorical_crossentropy"
  epochs: 150
  batch_size: 32
  learning_rate: 0.0001
  patience: 15
  validation_split: 0.2
  ...
```

Naming your project and setting paths can be found in the config file `configs/paths.yaml`. Name your trial projects using the section `Path for trial outputs`.
```
# ************** Path for trial outputs **************
trial_model_path: &trial_model_path 
  trial: "test-assessment"
  base: *out_path

trial_validate_path: &trial_validate_path
  trial: "test-assessment"
  base: *validate_out_path
```

It is not recommended to change any other paths, as they it is configured so directory creation is streamlined and sensible. 

## Running BRUIT

BRUIT currently only has two positional arguments, however, this framework is easily customizable and more can be added when needed.
Those commands are `preprocess` and `train`. 

The preprocessing takes in a directory that contains all of the files that need to be proprocessed. It will recursively go through each directory, using its name as a label, and preprocess the .wav audio files inside and store them in a preprocessed folder with the same label in the `/out` directory. 

The basic pipeline of preprocessing follows this:
```
.wav --> resample --> trim --> normalize --> pad/crop --> extract features (MFCC or Mel) --> classifier
```
However, there are my finer details within this pipeline which can be found within the included report.

An example command:
```
bruit preprocess -i /Data
```
You will see loading bars as the files are preprocessed, this can take some time. The preprocessed data is saved in the form of a numpy file `.npz`.

Once the files are preprocessed, training a model can begin. This framework allows three types of models to be trained currently (more can be easily added later). Those are a CNN, CNN-LSTM, or CNN-GRU. The choice of model can be set in the `parameters.yaml` file here:
```
72    model_type: "CNN" # CNN or CNN_Seq
```
Where CNN-Seq stands for Sequential CNN in which the LSTM or GRU is the sequential part. The choice betwween those can be found in the same file.
```
83     rnn_type: "gru"              # lstm or gru  
```
All CNN layers, filters, kernel sizes, dense layers, etc. can be changed within this section with ease. 

To train a model, the command is:
```
bruit train
```
An input file is optional! This usees the `paths.yaml` config to find the preprocessed file. If you want to train on separate file, use the optional input command.
```
bruit train -i /path/to/<optional-input>.npz
``` 

## Output Configuration

The output directory is setup as shown. The number of plots made determines on how many audio files and how many segments are found within each audio file.

```
out/
  └── <trial-name>/
       ├── <training_features.npz>
       ├── logs/
       ├── metadata/
       ├── figures/
       │   ├── CNN/
       │   |   ├── confusion-matrix.png
       │   |   └── training_history.png
       |   └── CNN_Seq/
       |       ├── confusion-matrix.png
       │       └── training_history.png
       ├── model/
       |    ├── CNN/
       |    └── CNN_Seq/
       └── preprocessed_audio/
           ├── artifact/
           |     ├── audio/
           |     |   ├── full/
           |     |   |   └── <audio.wav>
           |     |   └── segments/
           |     |       └── <segment_audio.wav>
           |     └── plots/
           |          ├── <segment_counts.png>
           |          ├── raw/
           |          |    └── <raw_waveform.png>
           |          ├── debug/
           |          |    └── <adaptive_lowcut_debug.png>
           |          ├── preprocessed/
           |          |    └── <waveform.png>
           |          ├── segments/
           |          |    └── <segment_waveform.png>
           |          └── spectrograms/
           |                 ├── raw/
           |                 |   └── <raw_spectrogram.png>
           |                 └── filtered/
           |                     └──<filtered_spectrogram.png>
           ├── healthy/
           |    ├── ...
           |    └── ...
           └── murmur/
                ├── ...
                └── ...
```