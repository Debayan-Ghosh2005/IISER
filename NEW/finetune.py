import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch

# STEP 1: Load your dataset
df = pd.read_csv('NEW/IMDB Dataset.csv')

# STEP 2: Convert labels
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df[['review', 'label']].dropna().reset_index(drop=True)

# STEP 3: Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42
)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

# STEP 4: Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set torch format
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# STEP 5: Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# STEP 6: Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_sentiment_results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
)

# STEP 7: Train using Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
