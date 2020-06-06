from dataset import GetData
from model import Deep_DenseOnly, metric
from tensorflow import keras
import numpy as np

class MainClass(GetData):
    def __init__(self,
                 DataName,
                 input_features,
                 wide,
                 activate):
        super().__init__(DataName, input_features)
        self.model = Deep_DenseOnly(len([v for v in input_features.values() if v==1]),
                                    wide,
                                    activate)
        self.train_y = np.where(self.train_y==-1, 0, 1)
        self.test_y = np.where(self.test_y==-1, 0, 1)
    
    def train(self,
              epoch,
              batch_size,
              learning_rate,
              loss="binary_crossentropy",
              optimizer=None):
        
        sgd = keras.optimizers.SGD(lr=learning_rate,
                                   decay=1e-6,
                                   momentum=0.9,
                                   nesterov=True)
        
        self.model.compile(optimizer=sgd, 
                           loss=loss,
                           metrics=[metric])
        for i in range(epoch):
            history = self.model.fit(self.train_x, self.train_y,
                                     batch_size=batch_size,
                                     epochs=1)
            
            if self.evaluate_x is not None:
                pred = self.model.predict(self.evaluate_x)
                self.pred_draw(pred, i)
        return history

    def test(self):
        score = self.model.evaluate(self.test_x,
                                    self.test_y,
                                    batch_size=100)
        print("TEST_LOSS={}, ACCURACY={}%".format(score[0], score[1]*100))

    
        
