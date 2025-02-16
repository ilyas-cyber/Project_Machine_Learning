from datasets import load_dataset
from transformers import AutoTokenizer

# Load your dataset (replace with your actual CSV file path)
dataset = load_dataset('csv', data_files={'train': 'xss_dataset.csv'})

# Inspect a few samples
print(dataset['train'][1])  # View the first sample
# print(dataset['train'][:4])  # Print the first 5 samples


# Load a tokenizer (distilBERT tokenizer for text preprocessing)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
