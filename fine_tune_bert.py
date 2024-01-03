import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

# Step 1: Loading the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Step 2: Tokenizing and preparing the dataset
def tokenize_batch(batch):
    # Tokenize a batch of text using the BERT tokenizer
    return tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)

# Example: Loading a sample dataset from Hugging Face
dataset = load_dataset("imdb")

# Converting labels to PyTorch tensors
labels = torch.tensor(dataset['train']['label'])

# Tokenizing the entire dataset
tokenized_dataset = dataset.map(tokenize_batch, batched=True)

# Combine tokenized inputs with labels
train_dataset = torch.utils.data.TensorDataset(
    tokenized_dataset['train']['input_ids'],
    tokenized_dataset['train']['attention_mask'],
    labels
)

# Step 3: Fine-tune the BERT model
# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset) * num_epochs)

# Training loop
for epoch in range(num_epochs):
    for batch in torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
        input_ids, attention_mask, labels = batch

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# Step 4: Save the fine-tuned model
model.save_pretrained('fine_tuned_model')
