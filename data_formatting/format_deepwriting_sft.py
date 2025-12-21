from datasets import load_from_disk

#Relative paths
train_data_dir = "data/deepwriting_train"
validation_data_dir = "data/deepwriting_validation"

# Output paths
formatted_train_dir = "data/deepwriting_formatted_train"
formatted_validation_dir = "data/deepwriting_formatted_validation"

MAX_SEQ_LENGTH = 1024 #Gemma 3 1B recommended max sequence length

def format_for_sft(row):
    """Converts DeepWriting dataset entries into required format for SFT training."""
    required_keys= ['prompt', 'solution']
    for key in required_keys:
        if key not in row or row[key] is None:
            raise ValueError(f"Missing required key: {key}")

    prompt = str(row['prompt']).strip()
    reasoning = str(row['solution']).strip()
    answer_placeholder = "See reasoning"
   
    text = (
        f"{prompt}\n"
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<answer>\n{answer_placeholder}\n</answer>"
    )

    #Output validation
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("Formatted text is invalid.")
    if "<reasoning>" not in text or "</reasoning>" not in text:
        raise ValueError("Formatted text missing reasoning tags.")
    if "<answer>" not in text or "</answer>" not in text:
        raise ValueError("Formatted text missing answer tags.")
    
    formatted_row = dict(row)
    formatted_row['text'] = text
    formatted_row['num_tokens'] = len(text.split())
    return formatted_row


def truncate_example(example, max_length=MAX_SEQ_LENGTH):
    """Truncates text to max_length tokens if necessary."""
    tokens = example["text"].split()
    if len(tokens) > max_length:
        example["text"] = " ".join(tokens[:max_length])
        example["num_tokens"] = max_length
    return example


def format_dataset(dataset):
    """Formats the entire dataset for SFT training."""
    #Formatting
    formatted_dataset = dataset.map(format_for_sft)

    #Truncation
    truncated_dataset = formatted_dataset.map(truncate_example)

    # #Optional filter
    # final_formatted_dataset = truncated_dataset.filter(
    #     lambda x: len(x["text"].split()) <= MAX_SEQ_LENGTH)
       
    #Compute and print stats
    total = len(truncated_dataset)
    truncated_count = sum (1 for x in truncated_dataset if x['num_tokens']== MAX_SEQ_LENGTH)
    retained_count = total - truncated_count

    print(f"Total samples: {total}")
    print(f"Retained samples within length: {retained_count} ({retained_count/total*100:.2f}%)")
    print(f"Truncated samples exceeding length:, {truncated_count} ({truncated_count/total*100:.2f}%)")


    return truncated_dataset



if __name__ == "__main__":
    # Load training and validation datasets
    train_dataset = load_from_disk(train_data_dir)
    validation_dataset = load_from_disk(validation_data_dir)

    print("Loaded training dataset length:", len(train_dataset))
    print("Loaded validation dataset length:", len(validation_dataset))

    # Format datasets for SFT
    formatted_train_dataset = format_dataset(train_dataset)
    formatted_validation_dataset = format_dataset(validation_dataset)


    print("Formatted training dataset length:", len(formatted_train_dataset))
    print("Formatted validation dataset length:", len(formatted_validation_dataset))

    print("Sample formatted training entry:\n", formatted_train_dataset[0])

    # Save formatted datasets to disk
    formatted_train_dataset.save_to_disk(formatted_train_dir)
    print(f"Formatted training set saved to {formatted_train_dir}")

    formatted_validation_dataset.save_to_disk(formatted_validation_dir)
    print(f"Formatted validation set saved to {formatted_validation_dir}")



