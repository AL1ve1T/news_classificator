## Documentation:
Project contains solution for ML Project assignment option 4.
The task of NN is to classify news dataset to 4 different categories: – b : business – t : science and technology – e : entertainment and –m : health. 

Entrypoint - `python app.py`

Project consists of 3 classes:
- Model - contains fields and methods used in training process
- Network - encapsulates NN
- Dataset - prepares data for train and test process

Both binary and multi-class classifications are implemented. Classes to classify are defined in file app.py :
 `dataset.prepare_for_cats(['cat1', 'cat2'])`
 Method gets list of labels, so here is the only code line to change, if you have to add more classes.
 
 Training, Testing and calculation of all metrics implemented in the method `Model.train()`
 
 ### Evaluation:
 It turned out to achieve high values of accuracy with a variety of combinations. Some examples of accuracies:
 
 Classes B and E; 100k dataset - 99% passed tests
 Classes B, E, T; 100k dataset - 97% passed tests
 Classes M, E, T; 100k dataset - 99% passed tests
 Classes M, E, T, B; 150k dataset - 94% passed tests

The most fast learning models contained classes: BE, TE, BM
The most slow learning models contained classes: BT
Other models learned equally well.