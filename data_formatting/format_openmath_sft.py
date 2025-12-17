from datasets import load_from_disk

#Relative paths
train_data_dir = "data/openmath_train"
validation_data_dir = "data/openmath_validation"

# Output paths
formatted_train_dir = "data/openmath_formatted_train"
formatted_validation_dir = "data/openmath_formatted_validation"

def format_for_sft(row):
    """Converts OpenMath dataset entries into required format for SFT training."""
    prompt = row['question']
    response = (
        "<reasoning>\n" 
        f"{row['generated_solution']}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{row['expected_answer']}\n"
        "</answer>"
    )
    return { "text": prompt + "\n" + response }

