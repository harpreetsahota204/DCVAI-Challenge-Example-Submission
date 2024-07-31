"""
This script is used to train the model for the project.

You should import your main functions from the data_curation.py script and use them to prepare the dataset for training.

The approved model is `yolov8m` from Ulytralytics. 

Your predictions must be in a label_field called "predictions" in the dataset.

You may pass your final selection of hyperparameters as keyword arguments in the load_zoo_model function. 

See here for more details about hyperparameters for this model: https://docs.ultralytics.com/modes/train/#train-settings

"""
from datetime import datetime
from math import log
import yaml

import fiftyone as fo
import fiftyone.utils.random as four
import fiftyone.utils.huggingface as fouh

from ultralytics import YOLO

from data_curation import prepare_dataset

def export_to_yolo_format(
    samples,
    classes,
    label_field="ground_truth",
    export_dir="./yolo_formatted",
    splits=["train", "val"]
):
    """
    Export samples to YOLO format, optionally handling multiple data splits.

    Args:
        samples (fiftyone.core.collections.SampleCollection): The dataset or samples to export.
        export_dir (str): The directory where the exported data will be saved.
        classes (list): A list of class names for the YOLO format.
        label_field (str, optional): The field in the samples that contains the labels.
            Defaults to "ground_truth".
        splits (str, list, optional): The split(s) to export. Can be a single split name (str) 
            or a list of split names. If None, all samples are exported as "val" split. 
            Defaults to None.

    Returns:
        None

    """
    if splits is None:
        splits = ["val"]
    elif isinstance(splits, str):
        splits = [splits]

    for split in splits:
        split_view = samples if split == "val" and splits == ["val"] else samples.match_tags(split)
        
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split
        )

def train_model(dataset, training_config):
    """
    Train the YOLO model on the given dataset using the provided configuration.
    """
    four.random_split(dataset, {"train": training_config['train_split'], "val": training_config['val_split']})

    export_to_yolo_format(
        samples=dataset,
        classes=dataset.default_classes,
    )

    model = YOLO(model="yolov8m.pt")

    # Check if epochs in train_params exceeds 50
    if 'epochs' in training_config['train_params'] and training_config['train_params']['epochs'] > 50:
        raise ValueError("Number of epochs cannot exceed 50. Please adjust the 'epochs' parameter in your training configuration.")

    results = model.train(
        data="./yolo_formatted/dataset.yaml",
        **training_config['train_params']
    )
    
    best_model_path = str(results.save_dir / "weights/best.pt")
    best_model = YOLO(best_model_path)

    return best_model


def run_inference_on_eval_set(eval_dataset, best_model):
    """
    Run inference on the evaluation set using the best trained model.

    Args:
        eval_dataset (fiftyone.core.dataset.Dataset): The evaluation dataset.
        best_model (YOLO): The best trained YOLO model.

    Returns:
        The dataset eval_dataset with predictions
    """
    eval_dataset.apply_model(best_model, label_field="predictions")
    eval_dataset.save()
    return eval_dataset


def eval_model(dataset_to_evaluate):
    """
    Evaluate the model on the evaluation dataset.

    Args:
        dataset_to_evaluate (fiftyone.core.dataset.Dataset): The evaluation dataset.

    Returns:
        the mean average precision (mAP) of the model on the evaluation dataset.
    """
    current_datetime = datetime.now()

    detection_results = dataset_to_evaluate.evaluate_detections(
        gt_field="ground_truth",  
        eval_key=f"evalrun_{current_datetime}",
        compute_mAP=True,
        )

    return detection_results.mAP()

def run():
    """
    Main function to run the entire training and evaluation process.

    Returns:
        None
    """
    with open('training_config.yaml', 'r') as file:
        training_config = yaml.safe_load(file)

    #train set
    curated_train_dataset = prepare_dataset(name="Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set")

    #public eval set
    public_eval_dataset = fouh.load_from_hub("Voxel51/DCVAI-Challenge-Public-Eval-Set")

    N = len(curated_train_dataset)
    
    best_trained_model = train_model(training_dataset=curated_train_dataset, training_config=training_config)
    
    model_predictions = run_inference_on_eval_set(eval_dataset=public_eval_dataset, best_model=best_trained_model)
    
    mAP_on_public_eval_set = eval_model(dataset_to_evaluate=model_predictions)

    adjusted_mAP = (mAP_on_public_eval_set * log(N))/N


if __name__=="__main__":
    run()
