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


def load_open_math():
    dataset = load_dataset("nvidia/OpenMathInstruct-1")

    #inspect the existing split of the OpenMathsInstruct-1 Dataset
    print("Original dataset splits:", dataset)
    print("Original dataset total length:", len(dataset['train'])+len(dataset['validation']))

    return dataset

def merge_open_math_splits(dataset):
    #Merge pre-split train and validation sets into a single training set
    full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    #Confirm the new dataset length
    print("New dataset total length", len(full_dataset))
    return full_dataset



if __name__ == "__main__":

    #load dataset
    dataset = load_open_math()

    #Merge train + validation splits
    full_dataset = merge_open_math_splits(dataset)

    #Inspection and verification 
    print("Column names in the dataset:",dataset['train'].column_names)
    
    #inpsect a sample from the training set
    print("Sample example from train:")
    print(dataset['train'][0])
    
    #Inspect a sample from the validation set
    print("Sample example from validation:")
    print(dataset['validation'][0])

    #Inspect a sample from the merged dataset
    print("Sample example from merged dataset:")
    print(full_dataset[0])

    #First example from the validation set to confirm merge
    print("First example from validation set in merged dataset:")
    print(full_dataset[len(dataset['train'])])




