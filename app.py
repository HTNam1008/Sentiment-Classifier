from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import gdown
import os

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

app = Flask(__name__)

device = torch.device("cpu")
model = SentimentClassifier(n_classes=3).to(device)

# Tải mô hình từ Google Drive nếu chưa tồn tại
model_path = 'phobert_fold5.pth'
if not os.path.exists(model_path):
    drive_file_id = '1-C07LYBp8zEf8Oovvw2dPdLCds_dIcRO'  # Thay thế bằng Google Drive ID thực tế
    gdown.download(f'https://drive.google.com/uc?id={drive_file_id}', model_path, quiet=False)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

class_names = ['negative', 'neutral', 'positive']

def infer(text, tokenizer, max_len=120):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, y_pred = torch.max(output, dim=1)

    return class_names[y_pred]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['sentence']
    prediction = infer(data, tokenizer)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
