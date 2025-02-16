import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

# Disable GPU (optional, for CPU-only training)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Step 1: Preprocess the CSV to filter malformed rows
input_file = "synthetic_xss_dataset.csv"
output_file = "cleaned_xss_dataset.csv"

try:
    # Load the CSV into pandas, ignoring malformed lines
    df = pd.read_csv(input_file, on_bad_lines="skip")
    # Save the cleaned dataset to a new file
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()

# Step 2: Load the cleaned dataset into the Hugging Face `datasets` library
try:
    dataset = DatasetDict({"train": Dataset.from_pandas(df)})
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()

# Step 3: Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    try:
        return tokenizer(example["text"], padding="max_length", truncation=True)
    except KeyError as e:
        print(f"Missing 'text' column in the dataset: {e}")
        return {}

try:
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Dataset tokenized successfully!")
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit()

# Step 4: Split dataset into train and test sets
try:
    train_test = tokenized_dataset['train'].train_test_split(test_size=0.2)
    print("Dataset split into train and test sets!")
except Exception as e:
    print(f"Error splitting dataset: {e}")
    exit()

# Step 5: Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # Binary classification (e.g., XSS vs. Non-XSS)
)

# Step 6: Define training arguments
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

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer,
)

# Step 8: Start training
print("Training started...")
try:
    trainer.train()
    print("Training completed!")
except Exception as e:
    print(f"Error during training: {e}")
