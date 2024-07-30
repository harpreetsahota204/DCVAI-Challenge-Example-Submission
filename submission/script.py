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

    This function exports the given samples to the YOLO format, which is commonly
    used for object detection tasks. It can handle single or multiple data splits
    (e.g., train, validation, test).

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

    Example:
        >>> import fiftyone as fo
        >>> dataset = fo.Dataset("my_dataset")
        >>> classes = dataset.default_classes
        >>> export_yolo_data(dataset, "/path/to/export", classes, splits=["train", "val", "test"])
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

def train_model(dataset):
    """
    Train the YOLO model on the given dataset.

    Args:
        dataset (fiftyone.core.dataset.Dataset): The dataset to train on.

    Returns:
        YOLO: The best trained model.
    """
    #split train set into train and validation, you can adjust these parameters
    four.random_split(dataset,{"train": 0.90, "val": 0.10})

    # Do not change the arguments here
    export_to_yolo_format(
        samples=dataset,
        classes=dataset.default_classes,
        )
    
    model = YOLO(model="yolov8m.pt")

    results = model.train(
        data="./yolo_formatted/dataset.yaml", #do not change this argument
        # you can pass your hyperparameters here, for example
        epochs=1,
        batch_size=8,
        imgzs=1280,
        # device="cuda",
        #so on and so forth
    )
    
    best_model_path = str(results.save_dir / "weights/best.pt") #do not change this argument

    best_model = YOLO(best_model_path)

    return best_model


def run_inference_on_eval_set(dataset, best_model):
    """
    Run inference on the evaluation set using the best trained model.

    Args:
        dataset (fiftyone.core.dataset.Dataset): The evaluation dataset.
        best_model (YOLO): The best trained YOLO model.

    Returns:
        None
    """
    dataset.apply_model(best_model, label_field="predictions")

def eval_model(eval_dataset):
    """
    Evaluate the model on the evaluation dataset.

    Args:
        eval_dataset (fiftyone.core.dataset.Dataset): The evaluation dataset.

    Returns:
        None
    """
    current_datetime = datetime.now()

    detection_results = eval_dataset.evaluate_detections(
        gt_field="ground_truth",  
        eval_key=f"evalrun_{current_datetime}",
        compute_mAP=True,
        )

    detection_results.mAP()


def run():
    """
    Main function to run the entire training and evaluation process.

    Returns:
        None
    """
    #train set
    curated_train_dataset = prepare_dataset(name="Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set")

    #public eval set
    public_eval_dataset = fouh.load_from_hub("Voxel51/DCVAI-Challenge-Public-Eval-Set")

    N = len(curated_train_dataset)
    
    best_trained_model = train_model(curated_train_dataset)

    mAP_on_public_eval_set = run_inference_on_eval_set(dataset=public_eval_dataset, best_model=best_trained_model)

    adjusted_mAP = (mAP_on_public_eval_set * log(N))/N



if __name__=="__main__":
    run()
