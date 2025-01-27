import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

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


