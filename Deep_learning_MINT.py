#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing necessary libraries and modules
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb

#Login to wandb to enable experiment tracking
wandb.login()

#Hyperparameters
input_size = 784  # Size of input features (e.g., 28x28 for MNIST)
hidden_size = 32  # Number of neurons
num_classes = 10  
num_epochs = 6
batch_size = 100  
learning_rate = 0.01 

#Data transformations for data augmentation
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Rotate images by a random angle between -10 and 10 degrees
    transforms.ColorJitter(brightness=0.2),  # Adjust brightness with a random factor
    transforms.ToTensor()
])

#Loading the MNIST dataset for training and testing
train_data = datasets.MNIST(root='data',
                           download=True, 
                            train=True, 
                            transform=data_transforms)

test_data = datasets.MNIST(root='data', 
                           download=True, 
                           train=False,
                           transform=data_transforms)


#Creating data loaders for training and testing data
train_loader=torch.utils.data.DataLoader(dataset=train_data,
                       batch_size=batch_size,
                       shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_data,
                      batch_size=batch_size,
                      shuffle=False)

#Checking the version of PyTorch
print(torch.__version__)

#Initializing a Wandb run to log experiment data
run = wandb.init(
    # Set the project where this run will be logged
    project="Log of diff DL models project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
    })


#Printing the number of training and testing samples
print(len(train_data))
print(len(test_data))


# In[4]:


import torch.nn.functional as F

#activasion models inside each class
#different layers in each class also 

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
         #MLP model
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.ReLU()  # Adding a ReLU activation after the second hidden layer
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.activation3 = nn.Tanh()  # Adding a Tanh activation after the third hidden layer
        self.l4 = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
         #Mlp model
        x = x.view(x.size(0), -1) #Flatten the input
        out = self.l1(x)
        out = self.activation1(out)
        out = self.l2(out)
        out = self.activation2(out)
        out = self.l3(out)
        out = self.activation3(out)
        out = self.l4(out)
        return out

class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNModel, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        #Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 100) #7x7 is the spatial size after two max-pooling layers
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        #CNN model
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool(out)
        
        out = out.view(out.size(0), -1) #Flatten the feature maps
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out
 


class OneDModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(OneDModel, self).__init__()
        #1-D convolutional
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1d = nn.ReLU()
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        
        #Calculating the input size 
        fc_input_size = 16 * (input_size - 2) // 2  

        # Fully connected layers for the 1-D model
        self.fc1d = nn.Linear(fc_input_size, 100)
        self.relu1d_2 = nn.ReLU()
        self.fc2d = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)#Reshape the input
        out = self.conv1d(x)
        out = self.relu1d(out)
        out = self.pool1d(out)
        out = out.view(out.size(0), -1)
        out = self.fc1d(out)
        out = self.relu1d_2(out)
        out = self.fc2d(out)
        return out


# In[5]:


#Check if a GPU is available; if not, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Change each hash to change model you want to run and log to wandb

#model = MLPModel(input_size, hidden_size, num_classes)
#model = CNNModel(input_size,num_classes)
model = OneDModel(input_size, num_classes)

#Defining the loss criterion
criterion = nn.CrossEntropyLoss()

#Change each hash to chnage wich optimizer you want to run and log to wandb

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8)


n_total_steps = len(train_loader)
total_steps = 0  # Initialize a total step counter

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass with original 2D image data
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward pass to get predictions
        outputs = model(images)
        
        #Calculating the loss
        loss = criterion(outputs, labels)
        
        #Performing backpropagation, and updating the model's parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loggin the loss and accuracy
        wandb.log({"Loss 1D_RMS": loss.item()}, step=total_steps)

        total_steps += 1  # Incrementing total step counter

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
        model.eval()
        val_loss = 0.0

    # Computing accuracy at the end of each epoch
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        
        acc = 100.0 * n_correct / n_samples

        # Loggin accuracy at the last step of each epoch
        wandb.log({"accuracy of 1D_RMS": acc}, step=total_steps)

    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    
#Saving the trained model
torch.save(model.state_dict(), 'trained_model.pth')

run.log_code()

wandb.finish()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




