import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load the image
image = cv2.imread('carParkImg.png')

# Convert the image to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image array to Torch tensor
image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float()

# Reshape and normalize the image
image_tensor = image_tensor.unsqueeze(0) / 255.0

# Run the image through the model
outputs = model(image_tensor)

# Retrieve the bounding box coordinates and labels
boxes = outputs[0]['boxes']
labels = outputs[0]['labels']

# Specify the label for cars
car_label = 3

# Process the detections
for box, label in zip(boxes, labels):
    if label == car_label:
        # Convert the box coordinates to integers
        box = box.int()
        
        # Extract the bounding box coordinates
        x, y, w, h = box
        
        # Convert the coordinates to integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Draw the bounding box rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the detected cars
cv2.imshow('Detected Cars', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
