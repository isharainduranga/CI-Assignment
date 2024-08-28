from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

# Load your saved model
model = torch.load('mymodel.pkl')
model.eval()  # Set the model to evaluation mode

def preprocess_image(image):
    # Define the preprocessing pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    image = Image.open(file).convert('L')
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
