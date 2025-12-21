from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np

TOKENIZER_NAME = "google/gemma-3-1b-it"

DATASETS = {
    "OpenMath Train": "data/openmath_formatted_train",
    "OpenMath Validation": "data/openmath_formatted_validation",
    "DeepWriting Train": "data/deepwriting_formatted_train",
    "DeepWriting Validation": "data/deepwriting_formatted_validation"
}


def inspect_token_lengths(dataset, tokenizer):
    lengths = []

    for example in dataset:
        tokens = tokenizer(
            example["text"],
            truncation=False,
            add_special_tokens=True
        )
        lengths.append(len(tokens["input_ids"]))

    lengths = np.array(lengths)

    print("\nToken length statistics:")
    print(f"Total samples: {len(lengths)}")
    print(f"Min tokens: {lengths.min()}")
    print(f"Mean tokens: {lengths.mean():.1f}")
    print(f"Median tokens: {np.median(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95)}")
    print(f"Max tokens: {lengths.max()}")

    return lengths


if __name__ == "__main__":

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for name, path in DATASETS.items():
        print(f"\nLoading dataset: {name} from {path}")
        dataset = load_from_disk(path)
        inspect_token_lengths(dataset, tokenizer)

    