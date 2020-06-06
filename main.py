from learning import MainClass

def main():
    DATA_NAME = "Circle"
    #DATA_NAME = "TwoGauss"
    #DATA_NAME = "Spiral"
    #DATA_NAME = "XOR"
    
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
    
    AI_sample = MainClass(DATA_NAME, FEATURES, NN_WIDEs, NN_ACTIVATEs)
    AI_sample.train(EPOCHs, BATCH_SIZE, LEARNING_RATE)
    AI_sample.test()

if __name__ == "__main__":
    main()
