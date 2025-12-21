from datasets import load_from_disk, Dataset
from langdetect import detect, DetectorFactory

# #Relative path
CLEANED_SAVE_DIR = "data/deepwriting_cleaned"

DetectorFactory.seed = 0  # For consistent language detection results

def inspect_dataset(dataset):
    """Provides summary statistics of empty fields, duplicates."""
    print("Starting clean_deepwriting.py")
    print("Dataset length:", len(dataset))

    #Count empty fields
    num_empty_prompts = sum(1 for x in dataset ['prompt'] if not x or str(x).strip() == "")
    num_empty_solutions = sum(1 for x in dataset ['solution'] if not x or str(x).strip() == "")
    print(f"Number of empty 'prompts' fields: {num_empty_prompts}")
    print(f"Number of empty solution' fields: {num_empty_solutions}")


def filter_english_text(dataset, prompt_field='prompt', solution_field='solution'):
    """Filters the dataset to retain only English text."""
    def is_english(row):
        try:
            prompt_text = str(row.get(prompt_field, "")).strip()
            solution_text = str(row.get(solution_field, "")).strip()

            if not prompt_text or not solution_text:
                return False
            return detect(prompt_text) == 'en' and detect(solution_text) == 'en'
        except Exception:
            return False

    english_rows = [row for row in dataset if is_english(row)]
    filtered_dataset = Dataset.from_list(english_rows)
    print(f"Filtered dataset to retain only English text. New length: {len(filtered_dataset)}")
    return filtered_dataset

def remove_empty_rows(dataset):
    """Removes rows with empty prompt or solution fields."""

    def is_row_valid(row):
        prompt = row.get("prompt")
        solution = row.get("solution")

        return (
            prompt is not None and str(prompt).strip() != "" and
            solution is not None and str(solution).strip() != ""
        )
    valid_rows = [row for row in dataset if is_row_valid(row)]
    cleaned_dataset = Dataset.from_list(valid_rows)

    print(f"Removed {len(dataset) - len(cleaned_dataset)} rows with empty 'prompt' or 'solution' fields.")
    return cleaned_dataset

def remove_duplicate_prompts(dataset):
    """Removes duplicate prompts from the dataset."""
    seen = set()
    unique_rows= []

    for row in dataset:
        prompt_key = str(row['prompt'])

        if prompt_key not in seen:
            unique_rows.append(row)
            seen.add(prompt_key)

    deduped_dataset = Dataset.from_list(unique_rows)
    print(f"Removed {len(dataset) - len(deduped_dataset)} duplicate prompts.")
    return deduped_dataset

def normalise_text(dataset):
    """Normalises text fields by stripping whitespace."""
    def normalize_row(row):
        if "prompt" in row and row["prompt"] is not None:
            row["prompt"] = str(row["prompt"]).strip()
        if "solution" in row and row["solution"] is not None:
            row["solution"] = str(row["solution"]).strip()
        return row

    normalized_dataset = dataset.map(normalize_row)
    print("Normalized text fields by stripping whitespace.")
    return normalized_dataset


def clean_deepwriting(dataset, save=False, save_dir=CLEANED_SAVE_DIR):
    """Cleans the DeepWriting dataset by removing duplicates, empty rows, and normalizing text, with optional saving."""
    print("Inspecting dataset before cleaning...")
    inspect_dataset(dataset)

    dataset = filter_english_text(dataset)
    dataset = remove_duplicate_prompts(dataset)
    dataset = remove_empty_rows(dataset)
    dataset = normalise_text(dataset)
  

    print("Final stats:")
    inspect_dataset(dataset)

    if save:
        dataset.save_to_disk(save_dir)
        print(f"Cleaned dataset saved to {save_dir}")

    return dataset



if __name__ == "__main__":

    print("Loading DeepWriting dataset from disk")
    dataset = load_from_disk("data/deepwriting_raw")

    print(dataset[0]['prompt'])
    print(type(dataset[0]['prompt']))

    cleaned_dataset = clean_deepwriting(dataset, save=True)

    #Final inspection
    print("\nFinal sample after cleaning:")
    for i in range(3):
        print(cleaned_dataset[i])
        print("------------")