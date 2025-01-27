from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_with_t5(text, model, tokenizer, max_input_length=512, max_output_length=50):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_output_length, min_length=10, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
