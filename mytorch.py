import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import function_wmmse_powercontrol as wf   # import our function file
import function_dnn_powercontrol as df     # import our function file

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

K = 10                      # number of users
num_H = 30000               # number of training samples
training_epochs = 100      # number of training epochs
trainseed = 0              # set random seed for training set
testseed = 7               # set random seed for test set

num_test = 500            # number of testing  samples

train_losses = []
valid_losses = []

# Problem Setup
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n'%(K, num_H, training_epochs))

# Generate Training Data
Xtrain, Ytrain, wtime = wf.generate_Gaussian(K, num_H, seed=trainseed)


# Define the Model
class PowerControlNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PowerControlNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PowerControlNet(input_size=K**2, output_size=K).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Data Preprocessing
# Data Transformation
X_train = Xtrain.T
Y_train = Ytrain.T

X_train_pt = torch.tensor(X_train, dtype=torch.float).to(device)
Y_train_pt = torch.tensor(Y_train, dtype=torch.float).to(device)

# Dataset Wrapping
train_dataset = TensorDataset(X_train_pt, Y_train_pt)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Dataset Splitting
num_train = len(X_train_pt)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))

np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)
valid_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=valid_sampler)


# Train and validate
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def validate(model, device, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(valid_loader.dataset)

for epoch in range(training_epochs):
    train_loss = train(model, device, train_loader, optimizer, criterion)
    valid_loss = validate(model, device, valid_loader, criterion)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

# Generate testing data
Xtest, Ytest, wtime_test = wf.generate_Gaussian(K, 10, seed=trainseed)
X_test = Xtest.T
Y_test = Ytest.T

# Data preprocessing
model.eval() 
X_test_pt = torch.tensor(X_test, dtype=torch.float).to(device)
Y_test_pt = torch.tensor(Y_test, dtype=torch.float).to(device)

# Prediction
start_time = time.time()
with torch.no_grad():
    predictions = model(X_test_pt).cpu().numpy() 
dnntime = time.time() - start_time

print("result:", predictions)
print("real:", Y_test_pt)

# Calculation
mse = mean_squared_error(Y_test_pt.cpu(), predictions)
mae = mean_absolute_error(Y_test_pt.cpu(), predictions)
r2 = r2_score(Y_test_pt.cpu(), predictions)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')


# Train loss and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')


#Bar graph of comparison
plt.figure(figsize=(12, 6))  # Specify the size of the figure
result_flatten = predictions.flatten()  # Flatten the predictions
y_test_flatten = Y_test.flatten()  # Flatten the true values
result_flatten = np.where(result_flatten > 0.5, 1, 0)
n = len(result_flatten)  # Get the number of data points
x_result = np.arange(n)  # Generate x coordinates for the predictions
x_y_test = x_result + 0.4   # Generate x coordinates for the true values, slightly offset for side-by-side display
plt.bar(x_result, result_flatten, width=0.4, label='Predict', alpha=0.6)  # Plot a bar graph of the predictions
plt.bar(x_y_test, y_test_flatten, width=0.4, label='Real', alpha=0.6)  # Plot a bar graph of the true values
plt.xlabel('Sample Index')  # Add an x-axis label
plt.ylabel('Value')  # Add a y-axis label
plt.title('Comparison of Predicted and Real Values')  # Add a title
plt.legend()  # Add a legend


# Calculate the counts of values close to 1 and close to 0
close_to_1 = np.sum(Ytrain >= 0.5)
close_to_0 = np.sum(Ytrain < 0.5)
# Prepare the data for the pie chart
labels = 'Close to 1', 'Close to 0'
sizes = [close_to_1, close_to_0]
colors = ['gold', 'lightcoral']
# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Proportion of Values Close to 1 and 0 in Ytrain')
plt.legend()  


# Confusion Matrix
Y_test_binarized = (y_test_flatten > 0.5).astype(int)
predictions_binarized = (result_flatten > 0.5).astype(int)
ConfusionMatrixDisplay.from_predictions(Y_test_binarized, predictions_binarized, normalize='true', display_labels=['0', '1'], cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')


# Performance Metrics
# Calculate confusion_matrix
tn, fp, fn, tp = confusion_matrix(Y_test_binarized, predictions_binarized).ravel()
# Calculate Performance
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# Plot the graph
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]
plt.figure()
bars = plt.bar(metrics, values)
plt.ylim(0, 1.1)
plt.title('Performance Metrics')
plt.ylabel('Value')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')


# Comparison on time
plt.figure(figsize=(5, 6))
plt.bar(['DNN Time', 'W Time'], [dnntime, wtime], color=['blue', 'red'])
plt.title('DNN Time vs W Time Comparison')
plt.ylabel('Time (in seconds)')
print("DNNTime:", dnntime)
plt.show()