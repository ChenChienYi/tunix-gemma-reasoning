"""
load_openmath.py

Utility functions to load and merge the OpenMathInstruct-1 dataset from Hugging Face.

Functions:
- load_open_math(): loads the dataset from HF Hub.
- merge_open_math_splits(dataset): merges the train and validation splits into a single dataset.

Author: Amita 
"""

# load datasets
from datasets import load_dataset, concatenate_datasets

#Relative path
DEFAULT_SAVE_DIR = "data/openmath_merged"

def load_open_math():
    """Load OpenMathInstruct-1 dataset from Hugging Face Hub."""
    dataset = load_dataset("nvidia/OpenMathInstruct-1")

    #inspect the existing split of the OpenMathsInstruct-1 Dataset
    print("Original dataset splits:", dataset)
    print("Original dataset total length:", len(dataset['train'])+len(dataset['validation']))

    return dataset

def merge_open_math_splits(dataset):
    """Merge pre-split train and validation sets into a single training set"""    
    full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])

    #Confirm the new dataset length
    print("New dataset total length", len(full_dataset))
    return full_dataset


def load_and_merge_openmath(save=False, save_dir=DEFAULT_SAVE_DIR):
    """Load and merge OpenMathInstruct-1 dataset, with optional saving."""
    dataset = load_open_math()
    full_dataset = merge_open_math_splits(dataset)

    if save:
        full_dataset.save_to_disk(save_dir)
        print(f"Merged dataset saved to {save_dir}")

    return full_dataset

if __name__ == "__main__":

    full_dataset = load_and_merge_openmath(save=True)

    #Inspection and verification 
    print("Merged dataset length:", len(full_dataset))
    print("Column names in the dataset:", full_dataset.column_names)
    print("First example in the merged dataset:", full_dataset[0])
   
    
    # #inpsect a sample from the training set
    # print("Sample example from train:")
    # print(dataset['train'][0])
    
    # #Inspect a sample from the validation set
    # print("Sample example from validation:")
    # print(dataset['validation'][0])

    # #Inspect a sample from the merged dataset
    # print("Sample example from merged dataset:")
    # print(full_dataset[0])

    # #First example from the validation set to confirm merge
    # print("First example from validation set in merged dataset:")
    # print(full_dataset[len(dataset['train'])])




