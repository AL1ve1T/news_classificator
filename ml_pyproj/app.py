#
#   File contains entrypoint for application
#

import csv
from dataset import Dataset
from model import Model

with open('./uci-news-aggregator.csv', 'rt') as f:
    data = csv.reader(f)
    data = list(data)
    
dataset = Dataset(data)
dataset.prepare_for_cats('b', 't')
model = Model(dataset)
model.train()

