# DCVAI-Challenge-Example-Submission

![visitors](https://visitor-badge.laobi.icu/badge?page_id=harpreetsahota204.DCVAI-Challenge-Example-Submission)

This project contains the code for training and evaluating a YOLOv8 model for the Voxel51 [Data-Centric Visual AI Challenge](https://huggingface.co/spaces/Voxel51/DataCentricVisualAIChallenge). 

For details about the competition and submission requirements, read the [`Submission information`](https://huggingface.co/spaces/Voxel51/DataCentricVisualAIChallenge) tab on the competition space.

Training dataset can be found [here](https://huggingface.co/datasets/Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set).

Public evaluation set can be found [here](https://huggingface.co/datasets/Voxel51/DCVAI-Challenge-Public-Eval-Set).

## Important Notes for Participants

1. **Implementing `data_curation.py`**: 

   - You need to implement the functions in the `data_curation.py` file. 

   - This file should contain your data preparation and curation strategies.

   - The main function to implement is `prepare_dataset()`, which will be called by the submission script.

2. **Do Not Modify Dataset Loading**:

   - Do not change the dataset loaded in the submission script.

   - A ValueError has been implemented to prevent loading different datasets.

   - The script is designed to work with the specified challenge dataset only.

3. **Configuring Hyperparameters**:

   - All hyperparameters for training the model should be set in the `training_config.yaml` file.

   - Modify this file to experiment with different training configurations.

   - The submission script will automatically use these parameters during training.

   - Important: The number of epochs is limited to a maximum of 50. If you set a higher value, the script will raise a ValueError.

   - Other key hyperparameters you may want to adjust include learning rate, batch size, and image size. Refer to the Ultralytics documentation for a full list of available parameters.

Remember, the focus of this challenge is on data-centric AI. Your efforts should be concentrated on improving the dataset quality and implementing effective data curation strategies in `data_curation.py`, rather than modifying the model architecture or training procedure.

## Project Structure

- `submission_script.py`: Main script for data preparation, model training, and evaluation.

- `data_curation.py`: Contains functions for dataset preparation .

- `training_config.yaml`: Configuration file for training hyperparameters.

- `requirements.txt`: List of Python dependencies.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/harpreetsahota204/DCVAI-Challenge-Example-Submission
   cd dcvai-challenge
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

You may need to install the following libraries:

```
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
```

3. Ensure you have access to the datasets as they are gated and you must request access:
   - Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set
   - Voxel51/DCVAI-Challenge-Public-Eval-Set

## Usage

1. Modify the `training_config.yaml` file to adjust training parameters as needed.

2. Run the main script:
   ```
   python submission_script.py
   ```

This script will:

- Prepare the training dataset

- Train the YOLOv8m model

- Run inference on the public evaluation set

- Compute and display the evaluation metrics

## Key Components

- `export_to_yolo_format()`: Exports the FiftyOne dataset to YOLO format.
- `train_model()`: Trains the YOLOv8m model using the prepared dataset.
- `run_inference_on_eval_set()`: Runs inference on the evaluation set.
- `eval_model()`: Evaluates the model performance.

## Configuration

The `training_config.yaml` file contains the following key parameters:

- `train_split`: Proportion of data to use for training.
- `val_split`: Proportion of data to use for validation.
- `train_params`: Dictionary of training parameters passed to the YOLO model.

Modify these parameters to experiment with different training configurations.

## Notes

- The model used is YOLOv8m from Ultralytics.

- Predictions are stored in a label field called "predictions" in the dataset.

