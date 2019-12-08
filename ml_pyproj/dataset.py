#
#   File contains class incapsulating dataset
#

import random

class Dataset:

    # Fields:
    # 	train_data, test_data (80%, 20%)
    # 	initial data (100%)
    
    def __init__(self, data):
    	
    	random.shuffle(data)
    	self.data = data
    	self.train_len = int(len(data) * 0.8)
    	self.test_len = int(len(data) * 0.2)
    	self.__prepare(self.data)

    def __prepare(self, data):
        
        self.train_data = data[:self.train_len]
        self.test_data = data[-self.test_len:]

    # Prepare Dataset for N categories
    def prepare_for_cats(self, cats):

        self.cats = cats

        self.train_data = list()
        self.test_data = list()

        temp_set = list()
        
        for row in self.data:
            if row[4] in self.cats:
                temp_set.append(row)
        self.__prepare(temp_set)

    # Drops train and test datasets to initial state
    # (without categories)
    def drop_default(self):
        self.__prepare(self.data)
        
        
