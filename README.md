# News Summarization Project

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Future Enhancements](#future-enhancements)
8. [Contributors](#contributors)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## Overview
This project focuses on building a news summarization system using advanced natural language processing techniques. The system aims to generate concise, meaningful summaries of news articles, providing readers with quick insights into lengthy news content. It leverages both abstractive and extractive summarization methods, demonstrating the superiority of the abstractive approach.

---

## Features
- Summarizes news articles from diverse sources (BBC News and Hindustan Times).
- Implements both abstractive and extractive summarization methods.
- Compares the performance of different models, including T5, LSTM, and Seq2Seq.
- Handles multi-domain content for robust summarization.

---

## Technologies Used
- **Programming Language:** Python
- **Libraries/Frameworks:**
  - TensorFlow
  - PyTorch
  - Hugging Face Transformers
  - NLTK
  - NumPy
  - Pandas
  - Matplotlib (for visualizations)
- **Models:**
  - T5 (Text-to-Text Transfer Transformer)
  - LSTM (Long Short-Term Memory Networks)
  - Seq2Seq (Sequence-to-Sequence Model)
- **Hardware:** GPUs for accelerated training

---

## Dataset
The dataset comprises news articles from:
- **BBC News:** Covering a variety of categories including politics, technology, and sports.
- **Hindustan Times:** Providing additional diversity in writing styles and topics.

The dataset was preprocessed to remove noise and irrelevant information, ensuring high-quality input for model training.

---

## Methodology
1. **Data Preprocessing:**
   - Text cleaning (removal of HTML tags, special characters, etc.).
   - Tokenization using NLTK.
   - Splitting data into training, validation, and test sets.

2. **Model Implementation:**
   - Trained T5, LSTM, and Seq2Seq models with varying hyperparameters (epochs, maximum sequence length).
   - Leveraged GPUs to expedite training.

3. **Evaluation:**
   - Metrics used: ROUGE (Recall-Oriented Understudy for Gisting Evaluation).
   - Compared abstractive and extractive methods, showcasing the enhanced readability and informativeness of abstractive summaries.

---

## Results
- **Abstractive Summarization:**
  - Generated coherent and meaningful summaries.
  - Outperformed extractive methods in terms of ROUGE scores.
- **Extractive Summarization:**
  - Limited to selecting exact sentences from the original text.
  - Often lacked context or fluency.

---

## Future Enhancements
- Incorporating real-time news feeds for live summarization.
- Extending support to more languages.
- Experimenting with larger, pre-trained transformer models like GPT.

---

## Contributors
- **Ravi Kanani**  
  [LinkedIn](https://www.linkedin.com/in/ravi-kanani/) | [Email](mailto:ravikanani2003@gmail.com)  
- **Om Joshi**  
  [LinkedIn](https://www.linkedin.com/in/om-joshi-0b9904284/) | [Email](mailto:om2003joshi@gmail.com)

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- Hugging Face for their transformer models.
- NLTK for preprocessing tools.
- TensorFlow and PyTorch for facilitating model training.
- BBC News and Hindustan Times for the dataset.

