# PytorchLightning porting of Facebook denoiser/demucs algorithm
This repository contains the straightforward porting of the pytorch demucs algorithm written by Facebook : https://github.com/facebookresearch/denoiser 

# Installation
First, install at least Python 3.6.9 (venv) with the following requirements :

    pytorch_lightning==1.0.4
    pandas==1.1.4
    tqdm==4.51.0
    torch==1.6.0
    sox==1.4.1
    numpy==1.19.3
    pesq==0.0.1
    pystoi==0.3.3
    torchaudio==0.6.0

# Dataset preparation
This repo uses the complete Valentini dataset https://datashare.is.ed.ac.uk/handle/10283/2791. When the dataset is downloaded, you first need to resample the audio samples to 16 kHz. You can use the script **sox_resampling.py** for that purpose. 
Finally, you need to generate the dataset files : train.csv, val.csv and test.csv. The script **prepare_valentini.py** will generate the files for you. You should obtain 3 .csv files with the following structure :
| clean_wav | noisy_wav  | n_samples |
|--|--|--|
| ?/clean_testset_wav/p232_001.wav | ?/noisy_testset_wav/p232_001.wav | 27861

# Training
The entry point for training is the file  **main_train_lightning.py**. It exposes the same parameters as the original implementation from Facebook. Hydra was removed from the pipeline. Slight modifications have been proposed for the Shift algorithm during data augmentation. A circular shifting is proposed to keep the full length of the audio samples inside a batch.
Comet is used as the default logger but feel free to use your personal taste. Use --deploy when you are ready to train your model of the full dataset.

    python main_train_lightning.py --pad --pesq --max_epochs 400 --glu --causal --normalize --n_worker 12 --gpus 1 --stft_loss --train egs/valentini/val.csv --valid egs/valentini/test.csv --bandmask 0.2 --remix --shift 5000 --dry 0.05 --deploy

