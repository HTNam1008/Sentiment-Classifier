from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Định nghĩa lớp SentimentClassifier
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
            return_dict=False # Dropout will error without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

# Tạo ứng dụng Flask
app = Flask(__name__)

# Load model và tokenizer
device = torch.device("cpu")
model = SentimentClassifier(n_classes=3).to(device)
model.load_state_dict(torch.load('models/phobert_fold5.pth', map_location=device))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

class_names = ['Tiêu cực', 'Bình thường', 'Tích cực']

def infer(text, tokenizer, max_len=120):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
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
