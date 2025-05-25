# DL-Convolutional Deep Neural Network for Image Classification

## AIM

To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## DESIGN STEPS

### STEP 1: 

Classify handwritten digits (0–9) using the MNIST dataset.

### STEP 2: 

Import required libraries such as TensorFlow/Keras, NumPy, and Matplotlib. Load the MNIST dataset using keras.datasets.mnist.load_data().

### STEP 3: 

Normalize the image pixel values (scale from 0-255 to 0-1). Reshape the images to match CNN input shape.

### STEP 4: 

Initialize a Sequential model. Add convolutional layers with activation (ReLU), followed by pooling layers. Flatten the output and add Dense layers. Use a softmax layer for classification.

### STEP 5: 

Compile the model with an optimizer (e.g., Adam), loss function (e.g., categorical crossentropy), and metrics (accuracy). Train the model using training data and validate using validation split or test data.

### STEP 6: 

Evaluate the model on test data and print accuracy. Plot training/validation loss and accuracy curves. Optionally, display a confusion matrix or sample predictions.

## PROGRAM

### Name: Sri Sai Priya S
### Register Number: 212222240103

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
train_data

test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)
test_data

type(train_data)

train_data[0]

type(train_data[0])

image,label = train_data[0]

image.shape
plt.imshow(image.reshape((28,28)),cmap='gray')
plt.show()

plt.imshow(image.reshape((28,28)),cmap='gist_yarg')
plt.show()

torch.manual_seed(101)  # for consistent results
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

from torchvision.utils import make_grid
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))
     
# First Batch
for images,labels in train_loader:
    break
     
images.shape
print('Labels: ', labels[:12].numpy())

im = make_grid(images[:12], nrow=12)
     
plt.imshow(np.transpose(im.numpy(),(1,2,0)));
plt.show()

class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[120,84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1],out_sz)

    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(101)
model = MultilayerPerceptron()
model

for param in model.parameters():
    print(param.numel())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

import time
start_time = time.time()

# Training
epochs = 10

# Trackers
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):

    trn_corr = 0
    tst_corr = 0
    for b,(X_train,y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train.view(100,-1))
        loss = criterion(y_pred,y_train)
        # 10 Neurons
        # [0.1, 0.0, .... 0.8]
        predicted = torch.max(y_pred.data,1)[1]   # print(y_pred.data)
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b%200 == 0:
            acc = trn_corr.item()*100/(100*b)
            print(f'Epoch{i}  batch {b}   loss: {loss.item()}  accuracy:{acc}')
    train_losses.append(loss)
    train_correct.append(trn_corr)
    with torch.no_grad():
        for b, (X_test,y_test) in enumerate(test_loader):
            y_val = model(X_test.view(500,-1))
            predicted = torch.max(y_val.data,1)[1]
            tst_corr += (predicted ==y_test).sum()
    loss = criterion(y_val,y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)
total_time = time.time() - start_time
print(f'Durations: {total_time/60} mins')
train_losses_np = [loss.detach().numpy() for loss in train_losses]
plt.plot(train_losses_np, label='Training loss')
plt.legend()
plt.show()
```

### OUTPUT

![image](https://github.com/user-attachments/assets/92bebb34-a2a3-4abf-8623-b26040f15ec9)

![image](https://github.com/user-attachments/assets/220b3734-1dc1-49ec-90b0-59b677628e5c)

![image](https://github.com/user-attachments/assets/b19066cc-f2a4-46a8-a223-97ebb5ce168e)

![image](https://github.com/user-attachments/assets/95f1558e-5d30-466c-a709-3f9408395059)

![image](https://github.com/user-attachments/assets/abb73998-f1b3-46b2-913c-5063bb420961)

![image](https://github.com/user-attachments/assets/56924979-39c5-4fb0-85ea-ea680d3a55b0)

![image](https://github.com/user-attachments/assets/df1cf296-9f2f-41c4-a1ac-3490f3f619d3)

![image](https://github.com/user-attachments/assets/89717764-25d7-4668-84b9-74857983378b)

![image](https://github.com/user-attachments/assets/058b2499-8807-42de-9e34-850e8f4b8bc7)

![image](https://github.com/user-attachments/assets/813e1b54-625e-459b-9c8c-30492c7f8dcb)

![image](https://github.com/user-attachments/assets/7db84563-d83d-4c16-a699-b0872ad6a36e)

![image](https://github.com/user-attachments/assets/e295cc1d-22eb-4a3b-af79-1bb7cc48c42f)

![image](https://github.com/user-attachments/assets/a83d62ba-2918-4204-8438-05cc59c98a23)

## RESULT

Thus the CNN model was trained and tested successfully on the MNIST dataset.
