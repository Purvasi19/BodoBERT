# BodoBERT for POS Tagging on Bodo Language Data

##  Project Overview

This project focuses on **Part-of-Speech (POS) Tagging for the Bodo language** using **BodoBERT**, a BERT-based language model specifically designed for Bodo text.

The objective of this project is to investigate how a pretrained Bodo language model can be fine-tuned for **token-level POS tagging** on a Bodo POS-tagged dataset written in **Devanagari script**.

The project covers the complete NLP pipeline, including:

* Bodo text preprocessing
* Tokenization using the BodoBERT tokenizer
* Conversion of POS tags into numerical labels
* Fine-tuning BodoBERT for token classification
* Handling subword tokenization and padding
* Model training and evaluation
* Prediction of POS tags for unseen Bodo sentences

---

##  Model Used

### BodoBERT

The model used in this project is:

**`alayaran/bodo-bert-mlm-base-article`**

BodoBERT is a pretrained Transformer-based language model developed for processing **Bodo language text**.

The pretrained model was originally trained using a **Masked Language Modeling (MLM)** objective. In this project, its pretrained language representations are adapted to the downstream task of **POS Tagging**.

For POS tagging, the BodoBERT Transformer is combined with a **token classification layer**, where each input token is assigned a POS tag.

### Model Architecture

```text
Bodo Sentence
      ↓
BodoBERT Tokenizer
      ↓
Input IDs + Attention Mask
      ↓
BodoBERT Transformer
      ↓
Contextual Token Representations
      ↓
Token Classification Layer
      ↓
POS Tag for Each Token
```

---

## Dataset

The dataset consists of **POS-tagged Bodo language sentences** written in **Devanagari script**.

Each sentence contains tokens along with their corresponding POS labels.

Example:

```text
Input:
बिदां फैनाय जादों ।

POS Tags:
NOUN VERB AUX PUNCT
```

The dataset contains approximately **6,000 sentences** before preprocessing.

After preprocessing and preparation for model training, the dataset was converted into a format suitable for Transformer-based token classification.

---

## Bodo Script

Bodo is an indigenous language of Northeast India and is primarily written using the **Devanagari script**.

Since BodoBERT is designed to process Bodo text, using a language-specific pretrained model allows the model to capture linguistic and contextual information specific to Bodo.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Load the Bodo POS-tagged dataset.
2. Separate the input tokens and corresponding POS labels.
3. Create a mapping between POS tags and numerical IDs.
4. Tokenize the input using the BodoBERT tokenizer.
5. Align the original POS labels with the generated subword tokens.
6. Apply padding and truncation.
7. Use `-100` for tokens that should be ignored when calculating the loss.

The `-100` label is particularly useful for special tokens and padding because PyTorch's cross-entropy loss ignores these positions.

---

## Tokenization

BodoBERT uses a subword-based tokenizer.

A single Bodo word may therefore be divided into multiple subword tokens.

For example:

```text
Original word
      ↓
BodoBERT Tokenizer
      ↓
Subword 1 | Subword 2 | Subword 3
```

Because POS tags are originally assigned at the **word level**, the labels need to be aligned with the resulting subword tokens.

This ensures that the model receives the correct POS supervision during training.

---

##  POS Tag Encoding

The POS tags were converted from textual labels into numerical IDs.

For example:

```text
NOUN → 0
VERB → 1
ADJ  → 2
ADV  → 3
...
```

The dataset contains approximately **36 POS tag classes**.

A corresponding `label2id` and `id2label` mapping was created so that the model could convert between numerical predictions and human-readable POS tags.

---

##  Fine-Tuning BodoBERT

The pretrained BodoBERT model was adapted for POS tagging using a **token classification architecture**.

Instead of predicting whether a token is masked, the final classification layer predicts the POS category of each token.

Conceptually:

```text
Pretrained BodoBERT
        ↓
Contextual Embeddings
        ↓
Linear Classification Layer
        ↓
36 POS Classes
        ↓
Predicted POS Tag
```

During training, the model compares its predicted POS tag with the actual POS tag and updates its parameters using backpropagation.

---

## Training Objective

The model is trained using **cross-entropy loss** for token classification.

For every token:

```text
Predicted POS Distribution
          ↓
Compare with
          ↓
Actual POS Label
          ↓
Cross-Entropy Loss
```

Padding and special-token positions are assigned the label `-100` so that they do not contribute to the loss.

---

## Evaluation

The trained BodoBERT model can be evaluated using token-level classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score

For POS tagging, **F1-score** is particularly useful because it evaluates how well the model predicts individual POS categories.

The evaluation also helps identify which POS categories are easier or more difficult for the model to recognize.

---

##  Prediction

After training, the model can be given an unseen Bodo sentence.

Example workflow:

```text
New Bodo Sentence
       ↓
BodoBERT Tokenizer
       ↓
Fine-Tuned BodoBERT
       ↓
POS Predictions
       ↓
Token + POS Tag
```

Example output:

```text
Token        POS
--------------------
बिदां        NOUN
फैनाय        VERB
जादों        AUX
।            PUNCT
```

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* BodoBERT
* NumPy
* Pandas
* Scikit-learn

---

##  Project Workflow

```text
Bodo POS Dataset
       ↓
Data Cleaning & Preprocessing
       ↓
POS Label Encoding
       ↓
BodoBERT Tokenization
       ↓
Subword Label Alignment
       ↓
Train / Validation / Test Data
       ↓
Fine-Tune BodoBERT
       ↓
Evaluate Model
       ↓
POS Prediction
```

---

##  Objective

The primary objective of this experiment is to study the effectiveness of a **pretrained Bodo-specific Transformer model** for POS tagging.

This experiment also demonstrates how pretrained language models can be adapted to downstream NLP tasks involving **low-resource Indian languages** such as Bodo.

---

## Future Improvements

Possible future improvements include:

* Increasing the size of the Bodo POS-tagged dataset
* Hyperparameter tuning
* Comparing BodoBERT with multilingual and Indic language models
* Using CRF on top of BodoBERT for structured sequence prediction
* Performing error analysis for individual POS categories
* Evaluating the model on additional Bodo NLP tasks
* Developing a complete Bodo NLP pipeline for downstream applications

---

##  Project Summary

This project demonstrates the use of **BodoBERT for Bodo POS Tagging** by fine-tuning a pretrained Transformer model on a POS-tagged Bodo dataset. The experiment provides an understanding of how language-specific pretrained models can be adapted for token-level NLP tasks and highlights the potential of Transformer architectures for **low-resource Indian language processing**.
