from datasets import load_from_disk

#Relative paths
train_data_dir = "data/openmath_train"
validation_data_dir = "data/openmath_validation"

# Output paths
formatted_train_dir = "data/openmath_formatted_train"
formatted_validation_dir = "data/openmath_formatted_validation"

def format_for_sft(row):
    """Converts OpenMath dataset entries into required format for SFT training."""
    required_keys= ['question', 'generated_solution', 'expected_answer']
    for key in required_keys:
        if key not in row or row[key] is None:
            raise ValueError(f"Missing required key: {key}")

    prompt = row['question']
    response = (
        "<reasoning>\n" 
        f"{row['generated_solution']}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{row['expected_answer']}\n"
        "</answer>"
    )
    text = prompt + "\n" + response

    #Output validatioon
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("Formatted text is invalid.")
    
    if "<reasoning>" not in text or "</reasoning>" not in text:
        raise ValueError("Formatted text missing reasoning tags.")
    
    if "<answer>" not in text or "</answer>" not in text:
        raise ValueError("Formatted text missing answer tags.")
    
    return {"text": text}

def format_dataset(dataset):
    """Formats the entire dataset for SFT training."""
    formatted_dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)
    return formatted_dataset

if __name__ == "__main__":
    # Load training and validation datasets
    train_dataset = load_from_disk(train_data_dir)
    validation_dataset = load_from_disk(validation_data_dir)

    print("Loaded training dataset length:", len(train_dataset))
    print("Loaded validation dataset length:", len(validation_dataset))

    # Format datasets for SFT
    formatted_train_dataset = format_dataset(train_dataset)
    formatted_validation_dataset = format_dataset(validation_dataset)

    assert formatted_train_dataset.column_names == ["text"]
    assert formatted_validation_dataset.column_names == ["text"]

    print("Formatted training dataset length:", len(formatted_train_dataset))
    print("Formatted validation dataset length:", len(formatted_validation_dataset))

    print("Sample formatted training entry:\n", formatted_train_dataset[0])

    # Save formatted datasets to disk
    formatted_train_dataset.save_to_disk(formatted_train_dir)
    print(f"Formatted training set saved to {formatted_train_dir}")

    formatted_validation_dataset.save_to_disk(formatted_validation_dir)
    print(f"Formatted validation set saved to {formatted_validation_dir}")