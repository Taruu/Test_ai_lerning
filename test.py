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

i=1
test = session.query(ticks).get(i)
print(test)
input()
meteor_data = [number for number in range(1, 34749 + 1)]
other_data = [number for number in range(1, 42811 + 1)]
random.shuffle(meteor_data)
random.shuffle(other_data)


metiors_test = [meteor_data[number] for number in range(31275, 31275 + 3474)]
ofther_data_test = [other_data[number] for number in range(38530, 38530 + 4281)]

len_metiors = len(meteor_data)
len_ofther_data_test = len(other_data)
for temp in range(len_ofther_data_test-1,38530,-1):
    other_data.pop(temp)
for temp in range(len_metiors-1,31274):
    meteor_data.pop(temp)

print(len(meteor_data), meteor_data)
print(len(other_data), other_data)
print(len(metiors_test),metiors_test)
print(len(ofther_data_test),ofther_data_test)



for meteor in meteor_data:
    metior_line = session.query(ticks).get(meteor)
