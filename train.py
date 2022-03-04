import torch                                    #Berk Limoncu 2243541
import torch.nn as nn
from model import Model1, Model2, Model3
from dataset import CIFARDataSet


def train(model, optimizer, train_data_loader,valid_data_loader, epochs, device_name):
    loss_function = nn.CrossEntropyLoss()
    model.train()

    print("epoch,step,loss")

    for epoch in list(range(epochs)):
        counter =0
        accum_train_loss = 0
        for i,(images, labels) in enumerate(train_data_loader,start=1):
            images,labels = images.to(device_name),labels.to(device_name)

            output = model(images)

            loss = loss_function(output,labels)

            accum_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter +=1

        model.eval()
        accum_val_loss=0
        with torch.no_grad():
            for j,(images,labels) in enumerate(valid_data_loader,start=1):
                images, labels = images.to(device_name), labels.to(device_name)
                output = model(images)
                accum_val_loss +=loss_function(output,labels).item()
        print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')



def run(epochs=100, layers=0, learning_rate=0.001, activation=256,function='relu'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = CIFARDataSet()

    if layers == 0:
        model = Model1()
    elif layers == 1:
        model = Model2(activation,function)
    else:
        model = Model3(activation,function)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train(model, optimizer, train_loader,val_loader, epochs, device)

    model.eval()
    with torch.no_grad():
        train_correct,train_total=0,0
        test_correct,test_total=0,0
        validation_correct,validation_total=0,0

        for images, labels in train_loader:
            images,labels = images.to(device), labels.to(device)
            output = model(images)

            _, pred_label = torch.max(output, dim=1)
            train_correct += (pred_label == labels).sum()
            train_total += labels.size(0)


        for images, labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)

            _, pred_label = torch.max(output, dim=1)
            test_correct += (pred_label == labels).sum()
            test_total += labels.size(0)


        for images, labels in val_loader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)

            _, pred_label = torch.max(output, dim=1)
            validation_correct += (pred_label == labels).sum()
            validation_total += labels.size(0)

        print("Training correctness: {:0.2f}".format(100*train_correct/train_total))
        print("Testing correctness: {:0.2f}".format(100*test_correct/test_total))
        print("TValidation correctness: {:0.2f}".format(100*validation_correct/validation_total))

        return 100*train_correct/train_total,100*test_correct/test_total,100*validation_correct/validation_total


if __name__=='__main__':
    seed = 1234
    torch.manual_seed(seed)

    learningRates = [0.01,0.001,0.0001]
    layerCounts = list(range(0,3))
    layerSizes = [256, 512, 1024]
    activationFunctions = ['relu','sigmoid','tanh']


    file = open("stats.txt","w")
    for lr in learningRates:
        for lc in layerCounts:
            for ls in layerSizes:
                for af in activationFunctions:
                    print(lr,lc,ls,af)
                    tr_c,te_c,va_c = run(100, lc, lr, ls, af)

                    written_text = "{} {} {} {} {} {} {}\n".format(str(lr),str(lc),str(ls),str(af),str(tr_c),str(te_c),str(va_c))

                    print(tr_c,te_c,va_c)
                    file.write(written_text)
    file.close()

