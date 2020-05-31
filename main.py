from learning import MainClass

def main():
    #DATA_NAME = "Circle"
    #DATA_NAME = "TwoGauss"
    #DATA_NAME = "Spiral"
    DATA_NAME = "XOR"
    
    NN_WIDEs = [5,5,5,1]
    NN_ACTIVATEs = ["tanh", "tanh", "tanh", "sigmoid"]

    EPOCHs = 100
    BATCH_SIZE = 100
    LEARNING_RATE = 0.03
    
    AI_sample = MainClass(DATA_NAME, NN_WIDEs, NN_ACTIVATEs)
    AI_sample.train(EPOCHs, BATCH_SIZE, LEARNING_RATE)
    AI_sample.test()

if __name__ == "__main__":
    main()
