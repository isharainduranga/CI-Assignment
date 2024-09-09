import os
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Define your CNN model architecture (replace with your actual architecture)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # New convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # New convolutional layer
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = None
        self.fc2 = nn.Linear(512, 256)       # New fully connected layer
        self.fc3 = nn.Linear(256, 10)        ## Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))   # Forward pass through the new convolutional layer
        x = self.pool(F.relu(self.conv4(x)))   # Forward pass through the new convolutional layer

        # Calculate the input size for the fully connected layer dynamically
        if self.fc1 is None:
            # Flatten the tensor and calculate the correct input size for fc1
            num_features = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(num_features, 512).to(x.device)

        # Adjust the view according to the actual output size
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))                # Forward pass through the new fully connected layer
        x = self.fc3(x)                        # Output layer
        return x


# Initialize Flask app
app = Flask(__name__)

# Initialize the model and load the weights
model = CNNModel()
model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')), strict=False)
model.eval()  # Set the model to evaluation mode

# Define the image transformation to match model input requirements
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values
])

@app.route('/')
def index():
    return render_template('./index.html')  # Serve your HTML file

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the image file
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_digit = predicted.item()

        # Return the prediction
        return jsonify({'predicted_digit': predicted_digit})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
