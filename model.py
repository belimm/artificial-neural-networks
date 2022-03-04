import torch                                #Berk Limoncu 2243541
import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):    # 0-Hidden Layer
    def __init__(self):
        super(Model1, self).__init__()
        self.layer1 = nn.Linear(32*32*1, 10)    #Gray Scale 32x32 img 32*32*1

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.layer1(x)


class Model2(nn.Module):    # 1-Hidden layer
    def __init__(self, num_neurons, function):
        super(Model2, self).__init__()
        self.layer1 = nn.Linear(32*32*1, num_neurons) #Gray Scale 32x32 img 32*32*1
        self.layer2 = nn.Linear(num_neurons, 10)    # Since the output shape of the network should match the number of classes, the number of units in the final layer should be 10.
        self.function = function

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        if self.function == "tanh":
            x = F.tanh(x)
        elif self.function == "sigmoid":
            x = F.sigmoid(x)
        elif self.function == "relu":
            x = F.relu(x)
        return self.layer2(x)


class Model3(nn.Module):    # 2-Hidden layer
    def __init__(self, num_neurons, function):
        super(Model3, self).__init__()
        self.layer1 = nn.Linear(32*32*1, num_neurons)    #Gray Scale 32x32 img 32*32*1
        self.layer2 = nn.Linear(num_neurons, num_neurons)
        self.layer3 = nn.Linear(num_neurons, 10)
        self.function = function

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        if self.function == "tanh":
            x = F.tanh(x)
        elif self.function == "sigmoid":
            x = F.sigmoid(x)
        elif self.function=="relu":
            x = F.relu(x)

        return self.layer2(x)


"""                                 #It was used to check the validity of the module
if __name__ == '__main__':

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    train_loader, val_loader, test_loader = dataset.CIFARDataSet()

    avg_train_loader ,avg_val_loader,avg_test_loader= 0.0,0.0,0.0
    total_train_loader,total_val_loader,total_test_loader = 0,0,0

    loss_function = nn.CrossEntropyLoss()


    for i,(images, labels) in enumerate(train_loader,start=1):
        pred = model(images)
        loss = loss_function(pred, labels)
        total_train_loader +=loss.item()

    print("Average loss: ",total_train_loader/i)

    avg_train_loader = total_train_loader/len(train_loader)



    exit()

    for images, labels in val_loader:
        pred = model(images)
        loss = F.nll_loss(pred,labels)
        total_val_loader +=loss.item()

    avg_val_loader = total_val_loader / len(val_loader)

    for images, labels in test_loader:
        pred = model(images)
        loss = F.nll_loss(pred, labels)
        total_test_loader += loss.item()
        print(pred)

    avg_test_loader = total_test_loader / len(test_loader)

    print("Avg train loader{} \n Avg val loader{} \n Avg test loader {}".format(avg_train_loader,avg_val_loader,avg_test_loader))
"""



