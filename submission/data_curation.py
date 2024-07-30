"""
This script is used to curate the data for the project. 

Implement your functions to to clean the data and prepare it for model training.

Note: the competition requires that you use FiftyOne for data curation and you are only allowed to
use the approaved dataset from the hub, Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set, which can 
be found here: https://huggingface.co/datasets/Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set
"""

import fiftyone as fo
import fiftyone.utils.huggingface as fouh

# Implement functions for data curation. below are just dummy functions as examples

def shuffle_data(dataset):
    """Shuffle the dataset"""
    return dataset.shuffle(seed=51)

def take_random_sample(dataset):
    """Take a sample from the dataset"""
    return dataset.take(size=10,seed=51)

def prepare_dataset(name = "Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set"):
    """Prepare the dataset for model training"""
    # Load the approved dataset from the hub
    # dataset = fouh.load_from_hub(name = "Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set", split="train")
    dataset = fo.load_dataset("Data-Centric-Visual-AI-Train-Set")
    dataset = shuffle_data(dataset)
    dataset = take_random_sample(dataset)
    curated_dataset = dataset.clone()
    return curated_dataset