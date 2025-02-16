from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import torch.nn.functional as F

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("xss_model/checkpoint-10")
tokenizer = DistilBertTokenizer.from_pretrained("xss_model/checkpoint-10")

# Set the model to evaluation model
model.eval()

# Test input
test_text = " this is not an xss <script>alert('xss') vulnerability"  # Example XSS payload
# test_text = "<script>alert('xss')</script>"  # Example XSS payload

inputs = tokenizer(test_text, return_tensors="pt")

# Get model outputs
with torch.no_grad():  # Disable gradient computation for inference
    outputs = model(**inputs)
    logits = outputs.logits

# Convert logits to probabilities
probs = F.softmax(logits, dim=-1)
prediction = torch.argmax(probs, dim=-1).item()

# Output prediction and confidence
print(f"Prediction: {'XSS Detected' if prediction == 1 else 'No XSS Detected'}")
print(f"Confidence: {probs[0][prediction].item():.4f}")

