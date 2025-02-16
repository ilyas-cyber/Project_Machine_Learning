import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Disable GPU (optional, for CPU-only training)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# File paths
dataset_file = "xss_uniqe_vectors.csv"

# Function to clean the dataset by removing malformed lines
def clean_dataset(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, header=None, names=["text", "label"])
        # Ensure every row has exactly two columns
        valid_df = df.dropna().reset_index(drop=True)
        print(f"Dataset cleaned: {len(df) - len(valid_df)} malformed rows removed.")
        valid_df.to_csv(file_path, index=False, header=False)
        return True
    except Exception as e:
        print(f"Failed to clean dataset: {e}")
        return False

# Function to train the model
def train_model():
    try:
        # Load the dataset
        dataset = load_dataset("csv", data_files={"train": dataset_file})
        print("Dataset loaded successfully!")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Tokenize the dataset
        def tokenize_function(example):
            return tokenizer(example["text"], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        print("Dataset tokenized successfully!")

        # Load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,  # Binary classification (e.g., XSS vs. Non-XSS)
        )

        # Split dataset into train and test sets
        train_test = tokenized_dataset["train"].train_test_split(test_size=0.2)
        print("Dataset split into train and test sets!")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./xss_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,  # Adjust if running out of memory
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_test["train"],
            eval_dataset=train_test["test"],
            tokenizer=tokenizer,
        )

        # Start training
        print("Training started...")
        trainer.train()
        print("Training completed!")
        return True

    except Exception as e:
        print(f"Error during training: {e}")
        return False


# Main loop: Clean dataset and retry training
while True:
    if not clean_dataset(dataset_file):
        print("Dataset cleaning failed. Please check the file manually.")
        break

    if train_model():
        print("Model trained successfully!")
        break
    else:
        print("Retrying after cleaning dataset...")
