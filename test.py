import csv
from itertools import islice
import numpy as np
import sqlalchemy
import glob
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pickle
from datetime import datetime
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Base = declarative_base()


engine = create_engine(r'sqlite:///C:\Users\VR_User\PycharmProjects\Test_ai_lerning\all_new_data_42811', echo=False)

Session = sessionmaker(bind=engine)
session = Session()



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(256,16,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,64,3)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class ticks(Base):
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    start = Column(Integer)
    end = Column(Integer)
    data = Column(sqlalchemy.types.BLOB)

    def __init__(self,name, start, end, data):
        self.start = start
        self.name = name
        self.end = end
        self.data = data

if __name__ == '__main__':
    meteor_data = [(number, torch.tensor([1])) for number in range(1, 34749 + 1)]
    other_data = [(number, torch.tensor([0])) for number in range(1, 42811 + 1)]
    random.shuffle(meteor_data)
    random.shuffle(other_data)

    metiors_test = [meteor_data[number] for number in range(31275, 31275 + 3474)]
    ofther_data_test = [other_data[number] for number in range(38530, 38530 + 4281)]

    len_metiors = len(meteor_data)
    len_ofther_data_test = len(other_data)
    for temp in range(len_ofther_data_test - 1, 38530, -1):
        other_data.pop(temp)
    for temp in range(len_metiors - 1, 31274):
        meteor_data.pop(temp)

    print(len(meteor_data), meteor_data)
    print(len(other_data), other_data)
    print(len(metiors_test), metiors_test)
    print(len(ofther_data_test), ofther_data_test)
    All_data_train = meteor_data
    All_data_train.extend(other_data)
    print(All_data_train)
    random.shuffle(All_data_train)
    print(All_data_train)


    net = Net()
    import torch.optim as optim
    print()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    from torch.autograd import Variable
    print("Выборка:",len(All_data_train))
    for epoch in range(2):
        print("start",epoch)
        running_loss = 0.0
        for i, data in enumerate(All_data_train, 0):
            inputs, labels = data[0],data[1].to(device)
            #print(inputs,labels)
            #print()
            opt.zero_grad()
            v = pickle.loads(session.query(ticks).get(inputs).data)["frames_x16"].to(device)
            v = Variable(v[None,...])
            outputs = net(v)
            loss = crit(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d, %5d]: loss = %.3f" % (epoch + 1, i + 1,
                                                  running_loss / 2000))
                running_loss = 0.0
    print("Train OK")
    torch.save(net.state_dict(), './cifar_net.path')




