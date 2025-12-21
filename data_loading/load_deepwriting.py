"""
load_openmath.py

Utility functions to load and merge the OpenMathInstruct-1 dataset from Hugging Face.

Functions:
- load_deepwriting(): loads and saves the dataset from HF Hub.

Note: Deep writing dataset doesnt specify an explicit license. It is used for research purposes only and not redistributed"""

from datasets import load_dataset

#Relative path
DEFAULT_SAVE_DIR = "data/deepwriting_raw"

def load_deepwriting(save= False, save_dir= DEFAULT_SAVE_DIR):
    """Load DeepWriting-20k dataset from Hugging Face Hub."""
    dataset_dict = load_dataset(
        "m-a-p/DeepWriting-20k",
        data_files = "deepwriting20k.parquet" 
    )

    dataset = dataset_dict['train']

    #inspect the Dataset
    print("Dataset loaded:DeepWriting-20k")
    print("Dataset length:", len(dataset))
    print("Column names in the dataset:", dataset.column_names)

    if save:
        dataset.save_to_disk(save_dir)
        print(f"Dataset saved to {save_dir}")

    return dataset

if __name__ == "__main__":
    
    dataset = load_deepwriting(save=True)
    
    #inspection
    print("First example in the dataset:", dataset[0])
    print(dataset[0].keys())
