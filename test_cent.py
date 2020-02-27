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
Base = declarative_base()


engine_ofther = create_engine(r'sqlite:////data/all_new_data_42811', echo=False)

Session_ofther = sessionmaker(bind=engine_ofther)
session_ofther = Session_ofther()

engine_meteors = create_engine(r'sqlite:////data/metiors_34749', echo=False)


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
    checkpoint = torch.load(filepath)
    print(checkpoint.keys())
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


meteors_test = [meteor_data[number] for number in range(31275, 31275 + 3474)]
ofther_data_test = [other_data[number] for number in range(38530, 38530 + 4281)]


test_data = meteor_data
test_data.extend(ofther_data_test)



net = Net()


name_file = input("Введите назание ")


model = load_checkpoint(name_file)
print()
