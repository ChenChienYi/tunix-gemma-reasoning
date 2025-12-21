from datasets import load_from_disk

# #Relative paths
clean_data_dir = "data/openmath_cleaned"
train_split_dir = "data/openmath_train"
validation_split_dir = "data/openmath_validation"

#split ratio
train_ratio = 0.1
random_seed = 42

def shuffle_and_split(dataset, train_ratio=0.1,seed=42):
    """Shuffles and splits the dataset into training and validation sets."""
    # Shuffle the dataset
    shuffled_dataset = dataset.shuffle(seed=seed)

    print("Splitting dataset...")
    split_dataset = shuffled_dataset.train_test_split(test_size=1 - train_ratio, seed=seed)

    train_dataset = split_dataset['train']
    validation_dataset = split_dataset['test']      

    return train_dataset, validation_dataset



if __name__ == "__main__":

    #load cleaned dataset
    dataset = load_from_disk(clean_data_dir)
    print("Loaded cleaned dataset length:", len(dataset))

    #Shuffle and split the cleaned dataset
    train_dataset, validation_dataset = shuffle_and_split(dataset, train_ratio, random_seed)

    assert len(train_dataset) + len(validation_dataset) == len(dataset)

    print("Training set length:", len(train_dataset))
    print("Validation set length:", len(validation_dataset))

    print(f"Train ratio: {len(train_dataset) / len(dataset):.3f}")
    print(f"Validation ratio: {len(validation_dataset) / len(dataset):.3f}")

    #Save the splits to disk
    train_dataset.save_to_disk(train_split_dir)
    print(f"Training set saved to {train_split_dir}")

    validation_dataset.save_to_disk(validation_split_dir)
    print(f"Validation set saved to {validation_split_dir}")

