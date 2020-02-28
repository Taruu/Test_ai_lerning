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

engine_ofther = create_engine(r'sqlite:///data/all_new_data_42811', echo=False)

Session_ofther = sessionmaker(bind=engine_ofther)
session_ofther = Session_ofther()

engine_meteors = create_engine(r'sqlite:///data/metiors_34749', echo=False)


Session_meteors = sessionmaker(bind=engine_meteors)
session_meteors = Session_ofther()



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
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    test_data = checkpoint["Test_data"]
    model.eval()
    return model,test_data




net = Net()


name_file = input("Введите назание ")
if len(name_file) < 2:
    name_file = "net-Thursday, 27. February 2020 05:31PM.pt"


model,test_data = load_checkpoint(name_file)
random.shuffle(test_data)
correct = 0
not_correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        inputs,labels = data
        if labels == torch.tensor([1]):
            v = pickle.loads(session_meteors.query(ticks).get(inputs).data)["frames_x16"]
        else:
            v = pickle.loads(session_ofther.query(ticks).get(inputs).data)["frames_x16"]
        v = Variable(v[None, ...])
        outputs = net(v)

        _, predicted = torch.max(outputs.data, 1)
        if labels == predicted:
            correct -=- 1
        else:
            not_correct -=-1
        #print(outputs,labels,predicted,"Верно предсказали? ",labels == predicted)

print(correct,not_correct)
print((correct/7755)*100)