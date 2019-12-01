#
#   File contains entrypoint for application
#

import csv
from dataset import Dataset

with open('./uci-news-aggregator.csv', 'rt') as f:
    data = csv.reader(f)
    data = list(data)
    
dataset = Dataset(data)
print(len(dataset.train_data))
print(len(dataset.test_data))

