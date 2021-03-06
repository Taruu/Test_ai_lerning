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
import pickle
from main import Net
from torch.autograd import Variable

Base = declarative_base()
print(torch.version.cuda)



mysql_enj = create_engine('mysql+mysqlconnector://mydb_user:root@localhost:3306/data_ai_learn', echo=False)

Session_mysql = sessionmaker(bind=mysql_enj)
session_mysql = Session_mysql()


engine_ofther = create_engine(r'sqlite:///data/all_new_data_42811', echo=False)

Session_ofther = sessionmaker(bind=engine_ofther)
session_ofther = Session_ofther()

engine_meteors = create_engine(r'sqlite:///data/metiors_34749', echo=False)


Session_meteors = sessionmaker(bind=engine_meteors)
session_meteors = Session_ofther()

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



def load_checkpoint(filepath):
    checkpoint = torch.load("Ai_train/"+filepath)
    print(checkpoint.keys())
    model = checkpoint['model']
    state = checkpoint['state_dict']
    #for parameter in model.parameters():
    #    parameter.requires_grad = False
    test_data = checkpoint["Test_data"]

    return model,state,test_data






name_file = input("Введите назание ")
if len(name_file) < 2:
    name_file = "net-03_03_20_09:04.pt"

model,state,test_data = load_checkpoint(name_file)
print(model)
print(state)
net = Net()
net.load_state_dict(state)
net.eval()

random.shuffle(test_data)
correct = 0
not_correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        inputs,labels = data
        if labels == torch.tensor([1]):
            data_temp = pickle.loads(session_mysql.query(mysql_metiors).get(inputs).data)["frames_x16"]
        else:
            data_temp = pickle.loads(session_mysql.query(mysql_other).get(inputs).data)["frames_x16"]
        v = Variable(data_temp[None, ...])
        outputs = net(v)

        _, predicted = torch.max(outputs.data, 1)
        if labels == predicted:
            correct -=- 1
        else:
            not_correct -=-1
        print('correct', correct, "not correct", not_correct)
        #print(outputs,labels,predicted,"Верно предсказали? ",labels == predicted)

print('correct',correct,"not correct",not_correct)
print((correct/7755)*100)