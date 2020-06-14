from learning import MainClass

def main():
    ################################################
    #Select one that you want model to learn from 4 datasets.
    ################################################
    #DATA_NAME = "Circle"
    #DATA_NAME = "TwoGauss"
    #DATA_NAME = "Spiral"
    DATA_NAME = "XOR"
    
    ################################################
    #Set HyperParameters.
    #NN_WIDES     : List contained numbers of nodes used in a layer. THe length means number of layers.
    #NN_ACTIVATEs : List contained types of activation functions. The length must be equal with NN_WIDES.
    #FEATURES     : Selection of input features. The "1" means the feature is inputed.
    #EPOCHs       : Total training epoch
    #BATCH_SIZE   : Number of inputed training data elements
    #LEARNING_RATE: The learning rate needed in model training
    ################################################
    NN_WIDEs = [5,5,5]
    NN_ACTIVATEs = ['tanh', 'tanh', 'tanh']
    FEATURES={'x1':1,
              'x2':1,
              'x12':0,
              'x22':0,
              'x1x2':0,
              'sin(x1)':0,
              'sin(x2)':0}
    EPOCHs = 100
    BATCH_SIZE = 100
    LEARNING_RATE = 0.03
    
    # Dataset preparation and model building.
    AI_sample = MainClass(DATA_NAME, FEATURES, NN_WIDEs, NN_ACTIVATEs)
    
    # Training model by training dataset. 
    AI_sample.train(EPOCHs, BATCH_SIZE, LEARNING_RATE)
    
    #Test model by test dataset. 
    AI_sample.test()

if __name__ == "__main__":
    main()
