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








import pickle
with open("Ai_all.pkl","rb") as file:
    vare = pickle.load(file)

print(vare.keys())

net = vare["net"]
print(vare["test_data"][0])
id_t = vare["test_data"][0]
print(id_t[0])
tsd = pickle.loads(session.query(ticks).get(id_t[0]).data)["frames_x16"]
with torch.no_grad():
    print(net(tsd))
