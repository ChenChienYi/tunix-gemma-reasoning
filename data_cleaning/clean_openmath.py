from datasets import load_from_disk, Dataset

# #Relative path
CLEANED_SAVE_DIR = "data/openmath_cleaned"

def inspect_dataset(dataset):
    """Provides summary statistics of empty fields, duplicates and rows with errors,before cleaning."""
    print("Starting clean_openmath.py")
    
    print("Dataset length:", len(dataset))
    #Count empty fields
    # only question, generated_solution, expected_answer are relevant for model learning
    num_empty_questions = sum(1 for x in dataset ['question'] if not x or str(x).strip() == "")
    num_empty_solutions = sum(1 for x in dataset ['generated_solution'] if not x or str(x).strip() == "")
    num_empty_answers = sum(1 for x in dataset ['expected_answer'] if not x or str(x).strip() == "")
    print(f"Number of empty 'question' fields: {num_empty_questions}")
    print(f"Number of empty 'generated solution' fields: {num_empty_solutions}")
    print(f"Number of empty 'expected answer' fields: {num_empty_answers}")

def count_duplicate_questions(dataset):
    """Counts duplicate questions in the dataset."""
    seen = set()
    duplicate_count = 0

    for q in dataset['question']:
        if q in seen:
            duplicate_count += 1
        else:
            seen.add(q)

    return duplicate_count

def remove_duplicate_questions(dataset):
    """Removes duplicate questions from the dataset."""
    unique_questions = set()
    unique_rows= []

    for row in dataset:
        if row['question'] not in unique_questions:
            unique_rows.append(row)
            unique_questions.add(row['question'])

    deduped_dataset = Dataset.from_list(unique_rows)
    print(f"Removed {len(dataset) - len(deduped_dataset)} duplicate questions.")
    return deduped_dataset

def remove_empty_rows(dataset):
    """Removes rows with empty question, generated_solution, or expected_answer fields."""

    required_fields = ['question', 'generated_solution', 'expected_answer'] 
    
    def is_row_valid(row):
        for field in required_fields:
            value = row.get(field)
            if value is None or str(value).strip() == "":
                return False
        return True
    clean_rows = [row for row in dataset if is_row_valid(row)]
    non_empty_dataset = Dataset.from_list(clean_rows)
    print(f"Removed {len(dataset) - len(non_empty_dataset)} rows with empty required fields.")
    return non_empty_dataset

def normalise_text(dataset):
    """Normalises text fields by stripping whitespace."""
    def normalize_row(row):
        for field in ['question', 'generated_solution', 'expected_answer']:
            if field in row and row[field] is not None:
                row[field] = str(row[field]).strip()
        return row

    normalized_dataset = dataset.map(normalize_row)
    print("Normalized text fields by stripping whitespace.")
    return normalized_dataset
                                                      

def run_inspection(dataset):
    #Inspect the first 3 examples
    print("First 3 examples in the dataset:")
    for i in range(3):
        print(dataset[i])
        print("------------")

    #count duplicate questions
    duplicate_count = count_duplicate_questions(dataset)
    print(f"Number of duplicate questions in the dataset: {duplicate_count}")

def print_stats(dataset, label):
    print(f"\n--- {label} ---")
    print("Dataset length:", len(dataset))

def clean_openmath(dataset, save=False, save_dir=CLEANED_SAVE_DIR):
    """Cleans the OpenMath dataset by removing duplicates, empty rows, and normalizing text, with optional saving."""
    print_stats(dataset, "Before Cleaning")

    dataset = remove_duplicate_questions(dataset)
    print_stats(dataset, "After Removing Duplicates")

    dataset = remove_empty_rows(dataset)
    print_stats(dataset, "After Removing Empty Rows")

    dataset = normalise_text(dataset)
    print_stats(dataset, "After Normalisation")

    if save:
        dataset.save_to_disk(save_dir)
        print(f"Cleaned dataset saved to {save_dir}")
    return dataset

if __name__ == "__main__":

    print("Loading merged dataset from disk")
    dataset = load_from_disk("data/openmath_merged")

    cleaned_dataset = clean_openmath(dataset, save=True)

    #Final inspection
    print("\nFinal sample after cleaning:")
    for i in range(3):
        print(cleaned_dataset[i])
        print("------------")