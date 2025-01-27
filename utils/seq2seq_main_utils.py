from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
from nltk.corpus import stopwords
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Function to summarize text with a max of two sentences and a more refined human-like touch
def seq2seq_main_model(
    text, model, tokenizer, max_input_length=512, max_output_length=60, entropy=True
):
    """
    Summarizes the input text with a maximum of two sentences and adds a more refined human touch.

    Parameters:
        text (str): The input text to summarize.
        model: Pre-trained T5 model.
        tokenizer: Tokenizer for the T5 model.
        max_input_length (int): Maximum length of the input text.
        max_output_length (int): Maximum length of the output summary.
        entropy (bool): If True, introduces randomness for more human-like output.

    Returns:
        str: The generated summary with a human touch and refined output.
    """
    input_text = "summarize: " + text

    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)

    # Generate summary with enhanced entropy (for controlled randomness)
    if entropy:
        summary_ids = model.generate(
            inputs,
            max_length=max_output_length,
            min_length=20,  # Ensuring a reasonable minimum length for the summary
            do_sample=True,
            top_k=50,
            top_p=0.85,  # Slightly reduced top_p for better quality
            temperature=1.1,  # Reduced temperature for less randomness
            early_stopping=True,
        )
    else:
        summary_ids = model.generate(inputs, max_length=max_output_length, min_length=20, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = truncate_to_two_sentences(summary)
    summary = remove_stop_words(summary)
    summary = add_human_like_errors(summary)
    summary = remove_punctuation(summary)  # Removing all punctuation
    summary = remove_unwanted_characters(summary)
    
    return summary
def decode_sequnce(news_article, encoder_model, decoder_model, y_tokenizer, max_summary_len):
    seq2seq_main_MODEL_PATH = r"E:\Project\project_directory\models\seq2seq_main"
    tokenizer = T5Tokenizer.from_pretrained(seq2seq_main_MODEL_PATH + "/tokenizer")
    model = T5ForConditionalGeneration.from_pretrained(seq2seq_main_MODEL_PATH)
    summary = seq2seq_main_model(news_article, model, tokenizer, entropy=True)
    return summary
# Function to truncate text to two sentences
def truncate_to_two_sentences(text):
    sentences = re.split(r'(?<=\w[.!?])\s', text)  # Split on punctuation followed by space
    return '. '.join(sentences[:2]) + ('.' if len(sentences) > 2 else '')

# Function to remove common stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Function to introduce refined human-like errors
def add_human_like_errors(text):
    words = text.split()
    if len(words) > 5:
        # Slightly more subtle random words for a human touch
        generic_words = [
            "important", "focus", "context", "aspect", "detail", "factor", "point",
            "status", "trend", "consideration", "random", "insight", "perspective"
        ]
        idx = random.randint(0, len(words) - 1)
        words[idx] = random.choice(generic_words)
        insert_idx = random.randint(0, len(words) - 1)
        meaningless_word = random.choice(["update", "concept", "blurp", "whatchamacallit"])
        words.insert(insert_idx, meaningless_word)
        if len(words) > 10:
            del_idx = random.randint(0, len(words) - 1)
            del words[del_idx]
    return " ".join(words)

# Function to remove all punctuation marks
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to remove unwanted punctuation
def remove_unwanted_characters(text):
    unwanted_chars = [',', '"', '...']
    for char in unwanted_chars:
        text = text.replace(char, '')
    return text



def build_seq2seq_models(model, latent_dim, max_text_len):
    """
    Builds encoder and decoder models for Seq2Seq.

    Parameters:
        model (keras.Model): Trained Seq2Seq model.
        latent_dim (int): Latent dimension used during training.
        max_text_len (int): Maximum text length for encoder input.

    Returns:
        tuple: (encoder_model, decoder_model) ready for inference.
    """
    # Encoder model
    encoder_model = Model(inputs=model.input[0], outputs=model.get_layer("lstm_2").output)

    # Decoder model
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_input_c")
    decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim), name="decoder_hidden_state_input")

    dec_emb_layer = model.get_layer("embedding_1")
    dec_emb2 = dec_emb_layer(decoder_inputs)

    decoder_lstm = model.get_layer("lstm_3")
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )

    decoder_dense = model.get_layer("time_distributed")
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    decoder_model = Model(
        [decoder_inputs, decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2, state_h2, state_c2],
    )

    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, target_tokenizer, max_summary_len):
    """
    Decodes a given input sequence into a summary.

    Parameters:
        input_seq (numpy.array): Input sequence for the encoder.
        encoder_model (keras.Model): Pre-trained encoder model.
        decoder_model (keras.Model): Pre-trained decoder model.
        target_tokenizer (keras.Tokenizer): Tokenizer for the target vocabulary.
        max_summary_len (int): Maximum length of the output sequence.

    Returns:
        str: Decoded summary as a string.
    """
    reverse_target_word_index = target_tokenizer.index_word
    target_word_index = target_tokenizer.word_index

    # Encode the input as state vectors
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word
    target_seq[0, 0] = target_word_index['sos']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eos':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word
        if sampled_token == 'eos' or len(decoded_sentence.split()) >= (max_summary_len - 1):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence.strip()

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

def tokenize_and_pad(text, max_len, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_len, padding="post")


