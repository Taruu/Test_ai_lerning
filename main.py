import csv
from itertools import islice
import numpy as np
import sqlalchemy
import glob
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import torchvision.transforms as transforms
import pickle
from datetime import datetime
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import datetime
import numpy
from torch.autograd import Variable
import torch.optim as optim
import time

Base = declarative_base()
#TODO выбор чтения данных

mysql_enj = create_engine('mysql+mysqlconnector://mydb_user:root@localhost:3306/data_ai_learn', echo=False)

Session_mysql = sessionmaker(bind=mysql_enj)
session_mysql = Session_mysql()

Session_mysql = sessionmaker(bind=mysql_enj)
session_mysql = Session_mysql()


engine_ofther = create_engine(r'sqlite:///data/all_new_data_42811', echo=False)

Session_ofther = sessionmaker(bind=engine_ofther)
session_ofther = Session_ofther()

engine_meteors = create_engine(r'sqlite:///data/metiors_34749', echo=False)


Session_meteors = sessionmaker(bind=engine_meteors)
session_meteors = Session_meteors() # тут была самая тупая ошибка!!!


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





class mysql_metiors(Base):
    __tablename__ = 'metiors'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    data = Column(sqlalchemy.types.BLOB)

    def __init__(self,id,name, data):
        self.id = id
        self.name = name
        self.data = data

class mysql_other(Base):
    __tablename__ = 'other'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    data = Column(sqlalchemy.types.BLOB)

    def __init__(self,name, data):
        self.name = name
        self.data = data


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(256,16,3)
        self.pool = nn.MaxPool2d(2,2)
        #self.conv2 = nn.Conv2d(16,64,3)
        self.input_layer = nn.Linear(256, 128)
        self.layer1 = nn.Linear(128,64)
        self.layer2 = nn.Linear(64,32)
        self.layer3 = nn.Linear(32,16)
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Linear(8, 4)
        self.end_layer = nn.Linear(4,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1, -1)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.end_layer(x)
        return x

def save_all(net,opt,test_data):
    save_to_piks = {"opt" : opt, "net" : net.state_dict() ,"test_data" : test_data}
    with open("Ai_train/Ai_all.pkl", "wb") as file:
        pickle.dump(save_to_piks,file)



if __name__ == '__main__':
    meteor_data = [(number, torch.tensor([1])) for number in range(1, 34748 + 1)]
    other_data = [(number, torch.tensor([0])) for number in range(1, 42811 + 1)]
    start = time.time()
    # for id,item in other_data:
    #     print(session_mysql.query(mysql_other).get(id).name)
    # end = time.time()
    # print(end - start)
    # input("end")
    random.shuffle(meteor_data)
    random.shuffle(other_data)

    meteors_test = [meteor_data[number] for number in range(31275, 31275 + 3474-1)]
    ofther_data_test = [other_data[number] for number in range(38530, 38530 + 4281)]


    test_data = meteors_test
    test_data.extend(ofther_data_test)

    len_metiors = len(meteor_data)
    len_ofther_data_test = len(other_data)



    for temp in range(len_ofther_data_test - 1, 38530, -1):
        other_data.pop(temp)
    for temp in range(len_metiors - 1, 31274):
        meteor_data.pop(temp)

    print(len(meteor_data), "meteor_data")
    print(len(other_data), "other_data")

    print(len(meteors_test), "meteors_test")
    print(len(ofther_data_test), "ofther_data_test")

    print(len(test_data),"test_data")
    print()

    All_data_train = meteor_data
    All_data_train.extend(other_data)

    print(All_data_train)

    random.shuffle(All_data_train)

    print(All_data_train)

    net = Net()

    print()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print("Выборка:",len(All_data_train))

    for epoch in range(2):
        print("start",epoch)
        running_loss = 0.0
        load_list = []
        load_transform_data =[]
        for i, data in enumerate(All_data_train, 0):
            if i % 20000 != 0 or i == 0 or (i == len(All_data_train)-1):
                load_list.append(data)
                continue
            for id_in_list in range(len(load_list)-1):
                inputs,label = load_list[id_in_list]
                if label == torch.tensor([1]):
                    data_temp = pickle.loads(session_mysql.query(mysql_metiors).get(inputs).data)["frames_x16"].to(device)
                else:
                    data_temp = pickle.loads(session_mysql.query(mysql_other).get(inputs).data)["frames_x16"].to(device)
                #print(data_temp)
                load_transform_data.append(( Variable(data_temp[None, ...]) , label ))
            print("train", i)
            #print(load_transform_data)
            #input()
            for i,data in enumerate(load_transform_data):
                inputs, labels = data[0], data[1].to(device)
                # print(inputs,labels)
                # print()
                #print(data)
                opt.zero_grad()

                # if labels == torch.tensor([1]):
                #     v = pickle.loads(session_meteors.query(ticks).get(inputs).data)["frames_x16"].to(device)
                # else:
                #     v = pickle.loads(session_ofther.query(ticks).get(inputs).data)["frames_x16"].to(device)

                #v = Variable(v[None, ...])
                outputs = net(inputs)
                loss = crit(outputs, labels)
                loss.backward()
                opt.step()

                running_loss += loss.item()
                # print(running_loss / 2000)
                #print(i % 2000)
                if i % 2000 == 1999:
                    print("[%d, %5d]: loss = %.3f" % (epoch + 1, i + 1,
                                                      running_loss / 2000))
                    running_loss = 0.0
            load_list.clear()
            load_transform_data.clear()

    print("Train OK")

    checkpoint = {'model': Net(),
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict(),
                  'Test_data': test_data}
    torch.save(checkpoint, "./Ai_train/net-{}.pt".format(datetime.datetime.now().strftime("%d_%m_%y_%H:%M")))

    #save_all(net,opt,test_data)
    #torch.save(net.state_dict(), './cifar_net.path')




