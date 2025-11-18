from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as T
from model import ImageClassifier


app = Flask(__name__)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model
model = ImageClassifier()
model_path = "cat_vs_dog_cnn_model.pth"
state = torch.load(model_path, map_location=device)
# If you saved state_dict: use load_state_dict
if isinstance(state, dict) and 'state_dict' in state:
model.load_state_dict(state['state_dict'])
else:
try:
model.load_state_dict(state)
except Exception:
# try loading whole model (less recommended)
model = torch.load(model_path, map_location=device)


model.to(device)
model.eval()


# Transforms
transform = T.Compose([
T.Resize((224, 224)),
T.ToTensor(),
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


labels = ["cat", "dog"]


def predict_image_bytes(image_bytes):
img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
x = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
out = model(x)
probs = torch.softmax(out, dim=1)
conf, idx = torch.max(probs, dim=1)
return labels[int(idx)], float(conf)


@app.route('/')
def index():
return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
if 'file' not in request.files:
return jsonify({'error': 'no file provided'}), 400
f = request.files['file'].read()
try:
label, conf = predict_image_bytes(f)
except Exception as e:
return jsonify({'error': str(e)}), 500
return jsonify({'label': label, 'confidence': conf})


if __name__ == '__main__':
app.run(host='0.0.0.0', port=7860)