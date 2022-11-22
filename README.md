# 12id-b-robo-steer

Automatic arm positioning prediction from previous user data via supervised learning.

## Model & Problem Definition

This project was created to predict the user-selected sensor position as the sensor is swept across a sample. When performing the sweep, only the luminosity reading is recorded. The model's goal is to predict the position a normal user would choose, a continuous value between -1 and 1, in order to potentially automate this process. 

The models were trained on around 900 samples, scraped from previous user data, with an 80:20 train test split. The data was augmented by shifting the reading positions to the left and right. Both the luminosity and arm position are normalized between -1 and 1 for better neural network performance. An MLP and ConvNet were applied to the data, but the MLP performed better in both accuracy and efficiency. 

## Installation and Usage

Install `pytorch` and `juypterlab` with respect to your current system. Then use `pip install -r requirements.txt` to install the generic requirements. Then install the library with `pip install .` from the project directory.

`manual_train.ipynb` and `hyperparameter_opt.py` can be used to train models. These models can then be used in inference via API's defined in the library component. `demo_use.ipynb` describes how to use these functions.
