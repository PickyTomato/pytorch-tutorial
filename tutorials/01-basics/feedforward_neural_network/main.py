import torch # loads pytorch.
import torch.nn as nn # loads Neural net module and calls nn.
import torchvision.datasets as dsets # load torchvision package which consists of popular datasets and calls dsets.
import torchvision.transforms as transforms # loads torchvision.transforms which are common image transform and calls transforms
from torch.autograd import Variable # load Variable from torch.autograd


# Hyper Parameters 
input_size = 784 # input data size
hidden_size = 500 # the number of hidden layers
num_classes = 10 # the number of class
num_epochs = 5 # the number of epochs 
batch_size = 100 # size of a batch
learning_rate = 0.001 # hyperparameter of learning rate

# MNIST Dataset 

# create training dataset
train_dataset = dsets.MNIST(root='./data', # Root directory where the data saved
                            train=True, # call the training dataset
                            transform=transforms.ToTensor(),  # transform as tensor form
                            download=True) # download the dataset

# create test dataset
test_dataset = dsets.MNIST(root='./data', # Root directory where the data saved
                           train=False, # call the test dataset
                           transform=transforms.ToTensor()) # transform as tensor form

# Data Loader (Input Pipeline)

# combine training set and a sampler
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # call the training dataset
                                           batch_size=batch_size, # the number of sample of a batch which is defined as hyperparameter
                                           shuffle=True) # shuffle the data to every epoch. if we don't shuffle the dataset, it will 
                                                         # be training with same data.
# combine test set and a sampler
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, # call the test dataset
                                          batch_size=batch_size, # same size of sample of a batch at training dataset
                                          shuffle=False) # Do not shuffle in every epoch. Because test set is fixed.

# Neural Network Model (1 hidden layer)

# create the Net class
class Net(nn.Module): # define the Net class. input is nn.Module
    def __init__(self, input_size, hidden_size, num_classes): # initialization method
        super(Net, self).__init__() # When initialize the method, it is method overiding of nn.Module(inherit nn.Modole to Net)
        self.fc1 = nn.Linear(input_size, hidden_size) # statement of fc1 is linear transformation to the incoming data
                                                      # in_feature = input_size, out_feature = hidden size
        self.relu = nn.ReLU() # statement of relu is applying Relu(x) function
        self.fc2 = nn.Linear(hidden_size, num_classes) # statement of fc2 is linear transformation to the incoming data
                                                       # in_feature = hidden_size, out_feature = num_classes
                                                       # It means use only one hidden layer.
    
    def forward(self, x): # define the forward process method
        out = self.fc1(x) # first linear transform the data (to 1st layer)
        out = self.relu(out) # Next using Relu function as activation function
        out = self.fc2(out) # Finally linear transform the result of first layer (to output layer)
        return out
    
net = Net(input_size, hidden_size, num_classes) # set the model

    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss() # Loss function is cross entropy function
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # Optimization algorithm is Adam. parameter can be called by model,
                                                                 # because it inherit nn.module. Learning rate, lr is learning_rate.

# Train the Model
for epoch in range(num_epochs): # loop 0 to num_epoch
    for i, (images, labels) in enumerate(train_loader): # i is index. loop 0 to the number of train_loader
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28)) # set image as 28 by 28 variables
        labels = Variable(labels) # set label variables
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer.
        outputs = net(images) # forward propagation.
        loss = criterion(outputs, labels) # calculate Loss using cross entropy.
        loss.backward() # backward propagation.
        optimizer.step() # optimization using Adam algorithm with simgle optimization step.
        
        if (i+1) % 100 == 0: # if a batch of data(# of 100 data) is learned, print as below.
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
            # (index of this epoch/whole epoch), (step of this data/the number of training set divided by batch size), (loss)

# Test the Model
correct = 0 # variable to check training set give right result. 
total = 0 # hh
for images, labels in test_loader: # loop 0 to the number of test_loader
    images = Variable(images.view(-1, 28*28)) # set image as 28 by 28 variables
    outputs = net(images) # forward propagation.
    _, predicted = torch.max(outputs.data, 1) # predicted data is max(output.data, 1)
    total += labels.size(0) # For calculating the number of whole test dataset.
    correct += (predicted == labels).sum() # if predicted number and label is the same, increase 1 to 'correct' 

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total)) # result of test using trained model

# Save the Model
torch.save(net.state_dict(), 'model.pkl') # Save the data
