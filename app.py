import os
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.t5_utils import summarize_with_t5
from utils.seq2seq_main_utils import decode_sequnce, preprocess_text, tokenize_and_pad
from utils.seq2seq_utils import decode_sequence, preprocess_text, tokenize_and_pad
from utils.seq2seq_utils import build_seq2seq_models, decode_sequence
from utils.seq2seq_main_utils import build_seq2seq_models, decode_sequence



# Paths to models
T5_MODEL_PATH = r"E:\Project\project_directory\models\t5_model"
SEQ2SEQ_MODEL_PATH = r"E:\Project\project_directory\models\seq2seq_model"
SEQ2SEQ_MAIN_MODEL_PATH = r"E:\Project\project_directory\models\seq2seq_main"

# Title and Description
st.title("News Summarization App")
st.write("Select a summarization model and provide a news article for summarization.")

# Model selection dropdown
model_choice = st.selectbox(
    "Choose a summarization model:",
    ("T5", "Seq2Seq","seq2seq_main"),
)

# Input text box
input_text = st.text_area("Enter the news article here:")

# Button to summarize
if st.button("Summarize"):
    if input_text.strip():
        if model_choice == "T5":
            # Load T5 model and summarize
            st.write("Using T5 model...")
            tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH + "/tokenizer")
            model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)
            summary = summarize_with_t5(input_text, model, tokenizer)
        
        elif model_choice == "Seq2Seq":
            # Load Seq2Seq model and summarize
            st.write("Using Seq2Seq model...")
            model = load_model(os.path.join(SEQ2SEQ_MODEL_PATH, "ns_model_60.h5"))
            
            # Load tokenizers and params
            with open(os.path.join(SEQ2SEQ_MODEL_PATH, "source_tokenizer.pkl"), "rb") as file:
                x_tokenizer = pickle.load(file)
            with open(os.path.join(SEQ2SEQ_MODEL_PATH, "target_tokenizer.pkl"), "rb") as file:
                y_tokenizer = pickle.load(file)
            with open(os.path.join(SEQ2SEQ_MODEL_PATH, "parameters.pkl"), "rb") as file:
                params = pickle.load(file)
            # Assume these are already loaded
            latent_dim = 300
            max_text_len = params["max_text_len"]
            max_summary_len = params["max_summary_len"]
            # Preprocess and tokenize input
            processed_text = preprocess_text(input_text)
            input_sequence = tokenize_and_pad(processed_text, max_text_len, x_tokenizer)

            # Generate summary
            input_sequence = tokenize_and_pad(processed_text, max_text_len, x_tokenizer)
            encoder_model, decoder_model = build_seq2seq_models(model, latent_dim, max_text_len)
            summary = decode_sequence(input_sequence, encoder_model, decoder_model, y_tokenizer, max_summary_len)
            print("Generated Summary:", summary)

        elif model_choice == "seq2seq_main":
             # Load Seq2Seq model and summarize
            st.write("Using Seq2Seq main model...")
            model = load_model(os.path.join(SEQ2SEQ_MAIN_MODEL_PATH, "ns_model_75.h5"))
            
            # Load tokenizers and params
            with open(os.path.join(SEQ2SEQ_MAIN_MODEL_PATH, "source_tokenizer.pkl"), "rb") as file:
                x_tokenizer = pickle.load(file)
            with open(os.path.join(SEQ2SEQ_MAIN_MODEL_PATH, "target_tokenizer.pkl"), "rb") as file:
                y_tokenizer = pickle.load(file)
            with open(os.path.join(SEQ2SEQ_MAIN_MODEL_PATH, "parameters.pkl"), "rb") as file:
                params = pickle.load(file)
            # Assume these are already loaded
            latent_dim = 300
            max_text_len = params["max_text_len"]
            max_summary_len = params["max_summary_len"]
            # Preprocess and tokenize input
            processed_text = preprocess_text(input_text)
            input_sequence = tokenize_and_pad(processed_text, max_text_len, x_tokenizer)

            # Generate summary
            input_sequence = tokenize_and_pad(processed_text, max_text_len, x_tokenizer)
            encoder_model, decoder_model = build_seq2seq_models(model, latent_dim, max_text_len)
            summary = decode_sequnce(input_text, encoder_model, decoder_model, y_tokenizer, max_summary_len)
            print("Generated Summary:", summary)
          
        st.subheader("Generated Summary:")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
