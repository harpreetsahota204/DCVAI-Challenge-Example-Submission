"""
This script is used to train the model for the project.

You should import your main functions from the data_curation.py script and use them to prepare the dataset for training.

The approved model is `yolov8m-coco-torch` from the FiftyOne Model Zoo.

Your predictions must be in a label_field called "predictions" in the dataset.

You may pass your final selection of hyperparameters as keyword arguments in the load_zoo_model function. 

See here for more details about hyperparameters for this model: https://docs.ultralytics.com/modes/train/#train-settings

"""

from data_curation import prepare_dataset

import fiftyone as fo
import fiftyone.zoo as foz

def export_to_yolo_format(
    samples,
    export_dir,
    classes,
    label_field="ground_truth",
    splits=None
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

# function to train model using yolo command line arguments

# function to apply model to dataset - should be able to do this with apply_model method




def run():

    dataset = prepare_dataset(name="Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set")
    
    export_to_yolo_format(
        samples=dataset,
        export_dir='yolo_data',
        classes=dataset.default_classes,
        splits=["train", "val"]
        )

    # Load the approved pre-trained model from the zoo
    model = foz.load_zoo_model(
        name="yolov8m-coco-torch",
        install_requirements=True,
        #example for how to pass hyperparameters
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        box=7.5
    )

    #apply_model to dataset
    dataset.apply_model(model, label_field="predictions")


if __name__=="__main__":
    run()
