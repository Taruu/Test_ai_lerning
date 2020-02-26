import torch
import torch.multiprocessing
import torchvision
import torchvision.transforms as transforms
import cv2, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.optim as optim
class Net(nn.Module):

    #TODO Подача две цветовых гаммы
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1000, 5) # Мы получили 3 картинки и затем поделили их на 6 потоков размером 5*5

        self.conv2 = nn.Conv2d(1000, 16, 5) # мы приянли 6 потоков и разделили их на 16 потков каждый из них это 5*5

        self.pool = nn.MaxPool2d(2, 2)

        #16 цветов на 5 на 5 пиксели???
        self.fc1 = nn.Linear(16 * 5 * 5, 128) # мы принмаем по умолчанию 5*5 но из за 16 потоков нужно домножить
        # 200 от первого слоя ко второму
        self.fc2 = nn.Linear(128, 64)
        self.fc2_2 =nn.Linear(64,32)
        self.fc2_3 = nn.Linear(32,16)
        # 84 от второго слоя
        self.fc3 = nn.Linear(16, 10)
        # резульятат это 84 входа которые завершают все 10 выходами класса
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))


        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc2_2(x))

        x = F.relu(self.fc2_3(x))

        x = self.fc3(x)

        return x


# P@ssw0rd

def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    # подачей на вход в pytorch
    run()

    # Нормирует все чтоб было от -1 до 1
    # Ибо оригинал это 0 до 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # загрузка набора данных
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)



    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True,
                                              num_workers=4)


    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False,
                                             num_workers=4)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck')
    dataiter = iter(trainloader)
    #
    images, labels = dataiter.next()

    grid = torchvision.utils.make_grid(images)

    grid = grid / 2 + 0.5
    n_grid = grid.numpy()

    #cv2.imshow("test", cv2.resize(np.transpose(n_grid, (1, 2, 0)),(200 * 4, 200)))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #cv2.waitKey(0)
    net = Net()


    crit = nn.CrossEntropyLoss(reduction="mean")
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(3):
        print("Now epoch",epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print(i,end=' ')
            inputs, labels = data

            opt.zero_grad()

            outputs = net(inputs)

            loss = crit(outputs, labels)

            loss.backward()

            opt.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print()
                print("[%d, %5d]: loss = %.3f" % (epoch + 1, i + 1,
                                                  running_loss / 2000))
                running_loss = 0.0



    print("Train OK")
    checkpoint = {'model': Net(),
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict()}
    torch.save(checkpoint, "./net-{}.pt".format(datetime.datetime.now().time()))