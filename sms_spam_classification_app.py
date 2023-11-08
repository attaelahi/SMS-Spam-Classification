import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# Specify the directory where you'll save your fine-tuned model
FINE_TUNED_MODEL_DIR = "./fine_tuned_sms_spam_model"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("wesleyacheng/sms-spam-classification-with-bert")
model = AutoModelForSequenceClassification.from_pretrained("wesleyacheng/sms-spam-classification-with-bert")

# Create a Streamlit app
st.title("SMS Spam Classification")

def classify_spam_or_ham(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Perform classification
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label
    predicted_label = "Spam" if outputs.logits[0][1] > outputs.logits[0][0] else "Not-Spam"

    return predicted_label

st.write("Single SMS Example:")


# Function to classify a single SMS
def classify_single_sms(text):
    if isinstance(text, str):  # Check if text is a string
        prediction = classify_spam_or_ham(text)
        st.write(f"SMS: {text}")
        st.write(f"Prediction: {prediction}")
        st.write("--------")
    else:
        st.warning("Skipping non-text data.")

# Main Streamlit code for CSV file upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with SMS messages:", type=["csv"])

if uploaded_file is not None:
    st.sidebar.write("Classifying SMS messages in the uploaded file...")
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')  # Specify the appropriate encoding
    except UnicodeDecodeError:
        st.sidebar.error("Error: Unable to decode the CSV file. Please make sure it is in the correct encoding.")
    else:
        # Allow the user to select the column containing SMS messages
        selected_column = st.sidebar.selectbox("Select the SMS column:", df.columns)

        if df[selected_column].dtype == "object":
            st.write("Classifications:")
            for sms_text in df[selected_column]:
                classify_single_sms(sms_text)
        else:
            st.sidebar.error("Selected column does not contain text data and cannot be tokenized.")

        st.sidebar.write("Classification completed!")

st.sidebar.write("Or classify a single SMS:")
user_input = st.sidebar.text_area("Enter an SMS message:")
if st.sidebar.button("Classify"):
    if user_input:
        classify_single_sms(user_input)
    else:
        st.sidebar.warning("Please enter an SMS message.")

st.write("Or fine-tune the model:")
if st.button("Fine-Tune Model"):
    if uploaded_file is not None and selected_column and df[selected_column].dtype == "object":
        # Use the data from the uploaded CSV file as the fine-tuning dataset
        custom_dataset = df[selected_column]

        # Specify your fine-tuning training arguments
        training_args = TrainingArguments(
            output_dir=FINE_TUNED_MODEL_DIR,
            overwrite_output_dir=True,
            per_device_train_batch_size=8,
            num_train_epochs=3,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=custom_dataset,
        )

        # Fine-tune the model
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained(FINE_TUNED_MODEL_DIR)
        tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
        st.write("Model has been fine-tuned and saved.")
    elif not uploaded_file:
        st.warning("Please upload a CSV file before fine-tuning.")
    elif not selected_column:
        st.warning("Please select the SMS column before fine-tuning.")
    else:
        st.warning("The selected column does not contain text data and cannot be used for fine-tuning.")
