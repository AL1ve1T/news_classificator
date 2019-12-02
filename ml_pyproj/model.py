#
#   File contains class incapsulating model
#

import torch
import numpy as np
import torch.nn as nn
from collections import Counter

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
    	super(Network, self).__init__()
    	self.layer_1 = nn.Linear(input_size,hidden_size, bias=True)
    	self.relu = nn.ReLU()
    	self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
    	self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)
    # accept input and return an output
    def forward(self, x):
    	out = self.layer_1(x)
    	out = self.relu(out)
    	out = self.layer_2(out)
    	out = self.relu(out)
    	out = self.output_layer(out)
    	return out

class Model:

    def __init__(self, dataset):    
        # Parameters
        self.plot_dir = './plot/'
        self.learning_rate = 0.01
        self.num_epochs = 3
        self.batch_size = 150

        self.dataset = dataset
        self.__prepare_nn(dataset)

        # Network parameters
        self.hidden_size = 100
        self.input_size = self.total_words
        self.num_classes = 2


    def __get_word_2_index(self, vocab):
        word2index = {}
        for i,word in enumerate(vocab):
            word2index[word.lower()] = i

        return word2index

    def __prepare_nn(self, dataset):

        vocab = Counter()
        data = [row[1] for row in self.dataset.data]

        for text in data: 
            for word in text.split(' '):
                vocab[word.lower()]+=1

        self.total_words = len(vocab)
        self.word2index = self.__get_word_2_index(vocab)

    def train(self):
        net = Network(self.input_size, self.hidden_size, self.num_classes)
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()  # This includes the Softmax loss function
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        losses = []
        # Train the Model
        for epoch in range(self.num_epochs):
            # determine the number of min-batches based on the batch size and size of training data
            total_batch = int(len(self.dataset.train_data)/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x,batch_y = self.__get_batch(self.dataset.train_data,i,self.batch_size)
                articles = torch.FloatTensor(batch_x)
                labels = torch.LongTensor(batch_y)

                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                outputs = net(articles)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.data)

                if (i+1) % 4 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                           %(epoch+1, self.num_epochs, i+1, 
                             len(self.dataset.train_data)/self.batch_size, loss.data))

        with open(self.plot_dir + self.dataset.cat1 + self.dataset.cat2, 'w') as f:
            f.write(str(losses))

    def __get_batch(self, df, i, batch_size):
        batches = []
        results = []
        data = list([row[1] for row in df])
        target = list([row[4] for row in df])
        # Split into different batchs, get the next batch 
        texts = data[i*batch_size:i*batch_size+batch_size]
        # get the targets 
        categories = target[i*batch_size:i*batch_size+batch_size]

        for text in texts:
            layer = np.zeros(self.total_words,dtype=float)

            for word in text.split(' '):
                layer[self.word2index[word.lower()]] += 1
            batches.append(layer)

        # We have 2 categories
        cat1 = self.dataset.cat1
        cat2 = self.dataset.cat2

        # 0 - category 1
        # 1 - category 2
        for category in categories:
            if category == cat1:
                index_y = 0
            else:
                index_y = 1
            results.append(index_y)

        # the training and the targets
        return np.array(batches),np.array(results)

