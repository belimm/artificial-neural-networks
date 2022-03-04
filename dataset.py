from torch.utils.data import DataLoader, random_split           #Berk Limoncu 2243541
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

def CIFARDataSet():

    transforms = T.Compose([
        T.ToTensor(),
        T.Grayscale(),
        T.Normalize((0.5,), (0.5,)),
    ])

    train_set = CIFAR10(root='CIFAR10', train=True, transform=transforms, download=True)

    train_set_length = int(0.8 * len(train_set))
    val_set_length = len(train_set) - train_set_length

    train_set, val_set = random_split(train_set, [train_set_length, val_set_length])
    test_set = CIFAR10(root='CIFAR10', train=False, transform=transforms, download=True)

    batch_size = 64

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


"""if __name__=='__main__':
    a,b,c = CIFARDataSet()      #It was used to check the validity of the module
"""





