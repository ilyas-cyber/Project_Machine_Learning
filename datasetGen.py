import random
import pandas as pd

# File paths
input_file_path = 'xss_dataset.csv'  # Replace with the path to your dataset
output_file_path = 'synthetic_xss_dataset.csv'  # Output file for the cleaned and synthetic dataset

try:
    # Load the existing dataset with error handling
    df = pd.read_csv(
        input_file_path,
        on_bad_lines='skip',  # Skip problematic rows
        quotechar='"',  # Handle quotes
        skipinitialspace=True,  # Ignore leading/trailing spaces
        encoding='utf-8',  # Ensure proper encoding
        engine='python'  # More flexible parser
    )
    
    # Check for required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input dataset must have 'text' and 'label' columns.")
    
    # Drop rows with missing or invalid data
    df = df.dropna(subset=['text', 'label'])  # Remove rows with missing values
    df['label'] = pd.to_numeric(df['label'], errors='coerce').dropna().astype(int)  # Ensure labels are integers

    # Ensure all strings are properly formatted
    df['text'] = df['text'].astype(str).str.strip()  # Strip leading/trailing spaces

    # Separate XSS and non-XSS examples
    xss_examples = df[df['label'] == 1]['text'].tolist()
    non_xss_examples = df[df['label'] == 0]['text'].tolist()

    # Check if we have sufficient examples
    if not xss_examples or not non_xss_examples:
        raise ValueError("Dataset does not contain sufficient XSS or non-XSS examples.")
except Exception as e:
    print(f"Error loading or validating dataset: {e}")
    exit()

# Define the size of the synthetic dataset
data_size = 1000  # Number of rows in the synthetic dataset

# Generate synthetic data
synthetic_data = [
    {
        "text": random.choice(xss_examples if random.random() > 0.5 else non_xss_examples),
        "label": 1 if random.random() > 0.5 else 0
    }
    for _ in range(data_size)
]

# Convert to a DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Save to a new CSV file
synthetic_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Synthetic dataset saved to {output_file_path}")
