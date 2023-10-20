
## Question(1)

For the question, i have used celeba dataset having 40 attributes of facial expressions of celebreties.
```bash
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
```
I took the subset of the whole data and labeled it with the selected 8 attributes, also converted the images in grey scale and reduce their size for easy computation.
``` bash
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import vgg16

# Load the CelebA annotations file
df = pd.read_csv('/content/extracted/list_attr_celeba.txt', delim_whitespace=True, header=1)

# Select the columns corresponding to the eight attributes to predict
selected_cols = ['Eyeglasses', 'Male', 'Smiling', 'Wearing_Hat', 'Wearing_Necklace', 'Wearing_Necktie', 'Wearing_Earrings', 'Wearing_Lipstick'] # choosed based on similarity


labels = df[selected_cols]

# Convert the labels to 0-1 binary values
labels = (labels == 1).astype(int)

# Select a random sample of the data
data = labels.sample(frac=0.20, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.index.values, data.values, test_size=0.2, random_state=42)

# Define the image size
img_size = (128,128)

# Load and preprocess the images
transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485], [0.229])])

X_train_processed = []
for img_file in X_train:
    img = Image.open('/content/extracted/img_align_celeba/' + str(img_file).zfill(6)).convert('RGB')
    img_tensor = transform(img)
    X_train_processed.append(img_tensor)
    
    
X_test_processed = []
for img_file in X_test:
    img = Image.open('/content/extracted/img_align_celeba/' + str(img_file).zfill(6)).convert('RGB')
    img_tensor = transform(img)
    X_test_processed.append(img_tensor)

# Convert the data to PyTorch tensors
X_train_tensor = torch.stack(X_train_processed)
X_test_tensor = torch.stack(X_test_processed)

y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)


```
The model architecture-
```bash

import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

# Define the ResNet18 backbone
resnet18_model = resnet18(pretrained=True)
resnet18_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

# Define the multi-task head
class MTH(nn.Module):
    def __init__(self, input_dim):
        super(MTH, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        # Forward pass through the multi-task head
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x.squeeze()

# Define the multi-task model
class MTM(nn.Module):
    def __init__(self, resnet18_model):
        super(MTM, self).__init__()
        # Use the ResNet18 backbone as the feature extractor
        self.backbone = nn.Sequential(*list(resnet18_model.children())[:-1])
        # Add a classifier to the model
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # Add multi-task heads to the model
        self.heads = nn.ModuleList([MTH(4096) for _ in range(8)])
    
    def forward(self, x):
        # Forward pass through the ResNet18 backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        # Forward pass through the classifier
        x = self.classifier(x)
        # Forward pass through the multi-task heads
        outputs = [head(x) for head in self.heads]
        return outputs

# Define the model and optimizer
model = MTM(resnet18_model)
optmz = optim.Adam(model.parameters(), lr=0.01)

# Define the loss function
loss_fn= nn.BCEWithLogitsLoss()


```
Training and evaluation of the model
```bash
# Train the model
num_epochs = 5
accuracy = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (inputs, labels) in enumerate(zip(X_train_tensor, y_train_tensor)):
        optmz.zero_grad()
        outputs = model(inputs.unsqueeze(0))
        losses = [loss_fn(output.unsqueeze(0), label.unsqueeze(0).float()) for output, label in zip(outputs, labels)]
        loss = sum(losses)
        loss.backward()
        optmz.step()
        epoch_loss += loss.item()
    epoch_loss /= len(X_train_tensor)
    print('Epoch [{}/{}], training_Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

    # Test the model
    model.eval()
    with torch.no_grad():
        right = 0
        entire = 0
        for inputs, labels in zip(X_test_tensor, y_test_tensor):
            outputs = model(inputs.unsqueeze(0))
            prds = [torch.sigmoid(output) > 0.5 for output in outputs]
            right += sum([prediction == label for prediction, label in zip(prds, labels)])
            entire += len(labels)
        accuracy = 100 * right / entire

print('Final Accuracy: {:.2f}%'.format(accuracy))


```
To evaluate Accuracy of each of the 8 tasks and plot
```bash
# Test the model and save accuracy of each task
model.eval()
with torch.no_grad():
    right = [0] * 8
    entire = [0] * 8
    accuracies = [[] for _ in range(8)]
    for inputs, labels in zip(X_test_tensor, y_test_tensor):
        otpts = model(inputs.unsqueeze(0))
        prds = [torch.sigmoid(output) > 0.5 for output in otpts]
        for task_idx, (prediction, label) in enumerate(zip(prds, labels)):
            right[task_idx] += int(prediction == label)
            entire[task_idx] += 1
            accuracy = 100 * right[task_idx] / entire[task_idx]
            accuracies[task_idx].append(accuracy)

# Print the accuracy of each task
for task_idx, accuracy in enumerate(accuracies):
    print('Final accuracy for Task {}: {:.2f}%'.format(task_idx+1, accuracy[-1]))

# Plot the accuracy of each task
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for task_idx in range(8):
    ax.plot(range(len(accuracies[task_idx])), accuracies[task_idx], label='Task {}'.format(task_idx+1))
ax.set_xlabel('Random Samples')
ax.set_ylabel('Accuracy')
ax.legend()
plt.show()
