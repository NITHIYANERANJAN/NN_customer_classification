# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1041" height="857" alt="image" src="https://github.com/user-attachments/assets/a8377ef1-2eb0-42a6-b8b8-fe62ee0f1980" />

## DESIGN STEPS

### STEP 1
Load and preprocess the dataset (handle missing values, encode categorical features, scale numeric data).

### STEP 2
Split the dataset into training and testing sets, convert to tensors, and create DataLoader objects.

### STEP 3
Build the neural network model, train it with CrossEntropyLoss and Adam optimizer, then evaluate with confusion matrix and classification report.

## PROGRAM
### Name: NITHIYANERANJAN S
### Register Number: 212223040136

```python


# Define Neural Network(Model1)
class NeuralNetwork(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = torch.nn.Linear(size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```python
# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
```
```python
# Initialize the Model, Loss Function, and Optimizer
cynthia_brain = NeuralNetwork(X_train.shape[1])
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cynthia_brain.parameters(), lr=0.001)
```
```python
# Train the model
train_model(cynthia_brain, train_loader, loss_fn, optimizer, epochs=50)
```
```python
# Evaluation
cynthia_brain.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = cynthia_brain(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())
```


## Dataset Information

<img width="911" height="675" alt="image" src="https://github.com/user-attachments/assets/36a338dd-e4e1-493a-9355-a67ae8418e3a" />


## OUTPUT

### Confusion Matrix

<img width="834" height="620" alt="image" src="https://github.com/user-attachments/assets/51cb4dad-9453-451e-9011-99a3d2538b17" />


### Classification Report

<img width="712" height="473" alt="image" src="https://github.com/user-attachments/assets/61083b1c-d5e3-4073-a279-424cb08c3628" />


### New Sample Data Prediction

<img width="578" height="119" alt="image" src="https://github.com/user-attachments/assets/23db4287-3457-4b56-afd9-bb5200d68d07" />


## RESULT
The neural network model was successfully built and trained to handle classification tasks.
