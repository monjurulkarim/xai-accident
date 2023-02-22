# Explainable Accident Anticipation
This is <b> partial implementation </b> of the paper, <a href="https://arxiv.org/pdf/2108.00273"> "Towards explainable artificial intelligence (XAI) for early anticipation of traffic accidents"</a> by Muhammad Monjurul Karim, Yu Li, and Ruwen Qin, accepted at Transportation Research Record.</p>

## Dataset
This code currently support CCD dataset for accident anticipation. 
> * Please download the extracted CCD dataset from [CCD-Dataset](https://bit.ly/3qXajsu). !ATTENTION! The training part of the data seems to be corrupted! I used only testing data with a 1/3 2/3 split in training.
> * (Optional) If you want to download the videos of CCD dataset, please refer to the [CarCrashDataset Official](https://github.com/Cogito2012/CarCrashDataset) repo for downloading and deployment. Then you can extract the frames using the code given in [scripts](https://github.com/monjurulkarim/xai-accident/tree/master/scripts).
> * To create human attention map this study collected gaze data for 100 videos taken from CCD test dataset. Video data along with their gaze information of 12 participants can be downloaded from [Here](https://drive.google.com/drive/folders/17F_wyVg5sQP-Vln93qHS17l-9AjEQsBG?usp=sharing)

## Folder structure
File structure:
Master folder should include
-> "xai-accident-fork" folder with cloned code. 
-> "data" folder with training and testing data.
-> "Crash-1500" annotation text file.

## Environment
Tested using conda package with pytorch and cuda.

toDO: Include list of needed  packages.
