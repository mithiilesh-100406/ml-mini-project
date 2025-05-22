import streamlit as st
import torch
from model import TransformerClassifier
import torch.nn.functional as F


VOCAB_SIZE = 10000  # must match training setup
SEQ_LEN = 128

# Dummy tokenizer: maps characters to integers (for demo)
def simple_tokenizer(text):
    tokens = [ord(c) % VOCAB_SIZE for c in text][:SEQ_LEN]
    if len(tokens) < SEQ_LEN:
        tokens += [0] * (SEQ_LEN - len(tokens))  # pad to fixed length
    return torch.tensor(tokens).unsqueeze(0)  # shape: (1, 128)

# Load model
model = TransformerClassifier(vocab_size=VOCAB_SIZE)
model.eval()

# UI
st.title("Transformer Classifier")
user_input = st.text_input("Enter your text:")

if user_input:
    input_tensor = simple_tokenizer(user_input)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()
    ans='positive' if pred_class==1 else 'negative'
    st.subheader("Model Output:")
    st.write("Raw logits:", output)
    st.write("Probabilities:", probs)
    st.write("Predicted Class:", ans)




