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

mysql_enj = create_engine('mysql+mysqlconnector://mydb_user:root@localhost:3306/data_ai_learn', echo=False)

engine_ofther = create_engine(r'sqlite:///data/all_new_data_42811', echo=False)

Session_ofther = sessionmaker(bind=engine_ofther)
session_ofther = Session_ofther()

engine_meteors = create_engine(r'sqlite:///data/metiors_new', echo=False)


Session_meteors = sessionmaker(bind=engine_meteors)
session_meteors = Session_meteors()

Session_mysql = sessionmaker(bind=mysql_enj)
session_mysql = Session_mysql()

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
i = 1
start = time.time()
while True:
    try:
        col = session_meteors.query(ticks).get(i)
        name = col.name
        data = col.data
        session_mysql.add(mysql_metiors(i,name, data))
        session_mysql.commit()
        i -= - 1
        if i % 100 == 0:
            print(i)
    except:
        print("end")
        break
end = time.time()
print(end - start)
print(i)
print("done")
