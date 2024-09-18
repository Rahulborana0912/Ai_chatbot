import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AdamW
from torch.nn import CrossEntropyLoss
import numpy as np
# Load the data
data = pd.read_csv('/content/final_data.csv')   # Replace with your actual file path

# Encode the labels
label_encoder = LabelEncoder()
data['id'] = label_encoder.fit_transform(data['id'])

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class QueryDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_len):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = str(self.queries[index])
        label = self.labels[index]  # Labels are now integers

        encoding = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'query_text': query,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create the dataset
train_dataset = QueryDataset(
    queries=train_data['user_query'].values,
    labels=train_data['id'].values,
    tokenizer=tokenizer,
    max_len=128  # Adjust max_len based on your data
)

test_dataset = QueryDataset(
    queries=test_data['user_query'].values,
    labels=test_data['id'].values,
    tokenizer=tokenizer,
    max_len=128
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['id'].unique()))

# Move the model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss().to(device)

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

EPOCHS = 3

for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        len(train_data)
    )
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Train loss {train_loss} accuracy {train_acc}')
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

test_acc, test_loss = eval_model(
    model,
    test_loader,
    loss_fn,
    device,
    len(test_data)
)
print(f'Test Accuracy: {test_acc}')
def predict(query, model, tokenizer, device):
    encoding = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    _, prediction = torch.max(output.logits, dim=1)

    return prediction.item()

# Example prediction
new_query = "Are there any known side effects of using pumpkin for digestive health?"
predicted_id = predict(new_query, model, tokenizer, device)
original_label = label_encoder.inverse_transform([predicted_id])[0]
print(f'Predicted ID: {original_label}')
# Save the model
model_save_path = "bert_model"
model.save_pretrained(model_save_path)

# Save the tokenizer
tokenizer_save_path = "bert_tokenizer"
tokenizer.save_pretrained(tokenizer_save_path)

# Optionally, save the label encoder
import pickle
label_encoder_save_path = "label_encoder.pkl"
with open(label_encoder_save_path, 'wb') as f:
    pickle.dump(label_encoder, f)
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

# Load the model
model = BertForSequenceClassification.from_pretrained(model_save_path)
model.to(device)  # Move to GPU if available

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)

# Load the label encoder
with open(label_encoder_save_path, 'rb') as f:
    label_encoder = pickle.load(f)
# Function to predict the class of a user query
def predict(query, model, tokenizer, device):
    encoding = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(output.logits, dim=1)

    return prediction.item()

# Loop to take input and provide prediction
while True:
    user_input = input("Enter your query (or type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting the chatbot.")
        break

    # Get prediction from the model
    predicted_id = predict(user_input, model, tokenizer, device)

    # Decode the predicted label
    original_label = label_encoder.inverse_transform([predicted_id])[0]
    print(f"Predicted Label: {original_label}")
import ipywidgets as widgets
from IPython.display import display

# Define the predict function again (for reference)
def predict(query, model, tokenizer, device):
    encoding = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(output.logits, dim=1)

    return prediction.item()

# Define the on-click function for the button
def on_button_click(b):
    user_query = input_text.value
    if user_query:
        predicted_id = predict(user_query, model, tokenizer, device)
        original_label = label_encoder.inverse_transform([predicted_id])[0]
        output_label.value = f"Predicted Label: {original_label}"

# Create input field, button, and output display
input_text = widgets.Text(
    value='',
    placeholder='Type your query',
    description='Query:',
    disabled=False
)

output_label = widgets.Label(value="Predicted Label will be shown here.")

predict_button = widgets.Button(
    description="Predict",
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to predict',
    icon='check'
)

# Set up the button click event
predict_button.on_click(on_button_click)

# Display the widgets
display(input_text, predict_button, output_label)
