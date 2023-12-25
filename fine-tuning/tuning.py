from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import numpy as np
#import evaluate

import warnings

warnings.filterwarnings("ignore")

from accelerate import Accelerator

accelerator = Accelerator()


print("LOAD DATASET")
dataset = load_dataset("yelp_review_full")

model_id = "bert-base-cased"

print("INIT TOKENIZER")
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("MAP DATASET")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

### model
print("INIT MODEL")
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5)

model = accelerator.prepare(
    model
)

### training
#metric = evaluate.load("accuracy")

#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return metric.compute(predictions=predictions, references=labels)

print("INIT TRAINER")
training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    #fp16=True
    #use_cpu=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    #compute_metrics=compute_metrics
)

print("TRAIN")
trainer.train()
#trainer.evaluate()