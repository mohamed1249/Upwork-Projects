# app.py
from flask import Flask, render_template, request
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from IPython.display import Image  # for displaying images


# Load the YOLOv5 model
yolo_model = torch.hub.load(r"C:\Users\LAPTOP WORLD\Downloads\best (1).pt",'custom', path='best.pt',force_reload=True,source='local', pretrained =False)

# Set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model.to(device).eval()

app = Flask(__name__)


# Load the YOLOv5 model
# yolo_model = torch.hub.load('.', 'custom', path=r"C:\Users\LAPTOP WORLD\Downloads\best (1).pt", source='local') 

# Set the device to use
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# yolo_model.to(device).eval()
# Define the image transformation
transform = T.Compose([
    T.ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded."

    image_file = request.files['image']
    image = Image.open(image_file)

    # Apply the transformation
    image_tensor = transform(image).unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():
        predictions = yolo_model(image_tensor)

        # Convert the PyTorch tensor to a PIL Image
        output_image = transform.ToPILImage()(image_tensor.squeeze(0))

        # Save or display the output image
        output_image.save('detected_'+image_file)

    # Process the predictions
    # (You need to adapt this part based on your model's output format)

    return "Vessel detected!"

if __name__ == '_main_':
    app.run(debug=True)