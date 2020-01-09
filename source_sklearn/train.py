from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.neural_network import MLPClassifier

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--first_hidden_layer_size', type=int, default=50, metavar='N',
                        help='the size of the first hidden layer (default: 50)')
    parser.add_argument('--second_hidden_layer_size', type=int, default=30, metavar='N',
                        help='the size of the second hidden layer (default: 30)')
    parser.add_argument('--third_hidden_layer_size', type=int, default=20, metavar='N',
                        help='the size of the third hidden layer (default: 20)')
 
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    ## TODO: Define a model 
    layer_sizes = (args.first_hidden_layer_size,args.second_hidden_layer_size,args.third_hidden_layer_size)
    model = MLPClassifier(hidden_layer_sizes=layer_sizes,batch_size=args.batch_size)

    
    
    ## TODO: Train the model
    
    model.fit(train_x,train_y)
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))