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

def prepare_dataset(name):
    """
    Prepare the dataset for model training.
    
    Args:
        name (str): The name of the dataset to load. Must be "Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set".
    
    Returns:
        fiftyone.core.dataset.Dataset: The curated dataset.
    
    Raises:
        ValueError: If the provided dataset name is not the approved one.
    
    Note:
        The following code block MUST NOT be removed from your submission:
        
        APPROVED_DATASET = "Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set"
        
        if name != APPROVED_DATASET:
            raise ValueError(f"Only the approved dataset '{APPROVED_DATASET}' is allowed for this competition.")
        
        This ensures that only the approved dataset is used for the competition.
    """
    APPROVED_DATASET = "Voxel51/Data-Centric-Visual-AI-Challenge-Train-Set"
    
    if name != APPROVED_DATASET:
        raise ValueError(f"Only the approved dataset '{APPROVED_DATASET}' is allowed for this competition.")
    
    # Load the approved dataset from the hub
    dataset = fouh.load_from_hub(name, split="train")
    
    # Implement your data curation functions here
    dataset = shuffle_data(dataset)
    dataset = take_random_sample(dataset)
    
    # Return the curated dataset
    curated_dataset = dataset.clone() 
    return curated_dataset