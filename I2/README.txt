CS 534 implementation problem two. 

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
matplotlib  3.1.1


commands to run code: 
      to train and validate data:   python Part1_Train.py
                                    python Part2_Train.py
                                    python Part3_Train.py
      to get test data predictions: python Part1_TestModel.py
                                    python Part2_TestModel.py
                                    python Part3_TestModel.py
                                    
more detailed description below----------------------------------------------------------------------------


problem description
-------------------------
a set of training data representing grey scale values of handwritten numbers 3 or 5 was provided (X)
using the training data, the actual number written was predicted (Y)
Online Perceptron, Average Perceptron, and Kernel Perceptron methods were tested to get the most accurate (W)


how to run this submission
-------------------------
structure of folders: within codes, the files needed to run this project are:
                               * Part1_Train.py (online perceptron)
                               * Part2_Train.py (average perceptron)
                               * Part3_Train.py (kernel perceptron)
                               * Part1_TestModel.py
                               * Part2_TestModel.py
                               * Part3_TestModel.py
  
        subcode needed for these files:
                               * data_reader.py (reads in and pre-processes assigned csv files)
         
 Results can be found in the output folder in csv format.
      
