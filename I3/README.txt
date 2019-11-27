CS 534 implementation problem three

team members
-------------------------
Derek Jackson
Meredith Leung
Dylan Sanderson

python and package information
-------------------------
python      3.7
numpy       1.17.2
pandas      0.25.1
pickle      (native to python)
matplotlib  3.1.1

commands to run code: 
      to train and validate data:   python part1.py
                                    python part2.py
                                    python part3.py
      to get test data predictions: python TestTree.py
            (note that the path to desired tree to test on should be specified)
                                    
more detailed description below----------------------------------------------------------------------------


problem description
-------------------------
a set of training data representing mushroom properties was provided (X)
using the training data, decision trees were built to predict whether each mushroom was poisonous or not (Y)
there were three parts:
      -part 1: decision tree
      -part 2: random forest
      -part 3: adaboost 


The predicition file for adaboost is located at '/output/pa3_test_predictions_p3.csv'