from transformers import AutoModelForSequenceClassification
from distilBERT import model  # Import the model from distilBERT.py

# Load pre-trained model with a classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2  # Binary classification (XSS or not)
)
