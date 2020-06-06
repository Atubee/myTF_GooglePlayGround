import numpy as np
import plygdata as pg
import matplotlib.pyplot as plt

def get_playground_dataset(Dname,
                           TRAINING_DATA_RATIO=0.8,
                           DATA_NOISE=0.0):
    
    if Dname == "Circle":
        data = pg.DatasetType.ClassifyCircleData
    elif Dname == "TwoGauss":
        data = pg.DatasetType.ClassifyTwoGaussData
    elif Dname == "Spiral":
        data = pg.DatasetType.ClassifySpiralData
    elif Dname == "XOR":
        data = pg.DatasetType.ClassifyXORData
    else:
        print('[E:dataset.py L18] Argument error [Dname]')
        exit()
    data = pg.generate_data(data, DATA_NOISE)
    data = pg.split_data(data,
                         training_size=TRAINING_DATA_RATIO)
    return data

def make_features(x1, x2, option):
    data = []
    if option['x1']:
        data.append(x1)
    if option['x2']:
        data.append(x2)
    if option['x12']:
        data.append(np.square(x1))
    if option['x22']:
        data.append(np.square(x2))
    if option['x1x2']:
        data.append(x1*x2)
    if option['sin(x1)']:
        data.append(np.sin(x1))
    if option['sin(x2)']:
        data.append(np.sin(x2))
    if not data:
        print('[E:dataset.py L39] Argument error [option]')
        exit()
    data = np.stack(data, -1)
    return data

def plot_setting(x1_min, x1_max, x2_min, x2_max):
    plt.ion()
    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_xlim(x1_min, x1_max)
    ax.set_xlim(x2_min, x2_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("TrainingData Plots and HeatMap on Pred Value")
    return fig, ax

class GetData:
    def __init__(self, DataName, input_features=None, evaluate=True):
        data = get_playground_dataset(DataName,
                                      TRAINING_DATA_RATIO=0.8,
                                      DATA_NOISE=0.0)
        self.train_x = data[0]
        self.train_y = data[1]
        self.test_x = data[2]
        self.test_y = data[3]

        x1_min, x1_max, x2_min, x2_max = -6, 6, -6, 6
        self.x1_max = x1_max
        self.x2_max = x2_max
        if evaluate:
            test_x1 = np.linspace(x1_min, x1_max, 100)
            test_x2 = np.linspace(x2_max, x2_min, 100)
            X1,X2 = np.meshgrid(test_x1, test_x2)
            e_data = np.stack((X1,X2), -1)
            self.X1, self.X2 = X1, X2
            self.evaluate_x = np.reshape(e_data, (-1,2))
        else:
            self.evaluate_x = None
            
        self.fig, self.ax = plot_setting(x1_min, x1_max, x2_min, x2_max)
        self.mpl_draw()

        if input_features is not None:
            self.train_x = make_features(self.train_x[:,0],
                                        self.train_x[:,1],
                                        input_features)
            self.test_x = make_features(self.test_x[:,0],
                                        self.test_x[:,1],
                                        input_features)
            self.evaluate_x = make_features(self.evaluate_x[:,0],
                                            self.evaluate_x[:,1],
                                            input_features)
        
    def mpl_draw(self):
        mask = self.train_y[:,0]==1
        self.ax.scatter(self.train_x[mask, 0],
                        self.train_x[mask, 1],
                        zorder=2,
                        edgecolors="white",
                        label="1th class")
        mask = self.train_y[:,0]==-1
        self.ax.scatter(self.train_x[mask, 0],
                        self.train_x[mask, 1],
                        zorder=2,
                        edgecolors="white",
                        label="2th class")
        self.ax.legend(loc='upper right')
        plt.draw()
        plt.pause(1)

    def pred_draw(self, pred, epoch):
        """
        x = self.evaluate_data[:, 0]
        y = self.evaluate_data[:, 1]
        sc = self.ax.scatter(x,
                             y,
                             s=10,
                             c=pred,
                             cmap='coolwarm',
                             zorder=1)
        """
        data = pred.reshape(100,100)
        sc = self.ax.pcolormesh(self.X1,
                                self.X2,
                                data,
                                vmin=0,
                                vmax=1,
                                cmap="coolwarm_r")
        cn = self.ax.contour(self.X1,
                             self.X2,
                             data,
                             levels=[0.5],
                             colors='g')
        tx = self.ax.text(self.x1_max,
                          self.x2_max,
                          str(epoch+1)+"Epoch",
                          size = 10,
                          color = "black")
        
        cbar = plt.colorbar(sc)
        cbar.set_label('Predicted value') 
        plt.draw()
        plt.pause(1e-10)
        cbar.remove()
        sc.remove()
        cn.collections[0].remove()
        tx.remove()        
        
        
if __name__ == "__main__":
    input_features={'x1':1,
                    'x2':0,
                    'x12':0,
                    'x22':0,
                    'x1x2':0,
                    'sin(x1)':0,
                    'sin(x2)':0}
    num = [k for k, v in input_features.items() if v == 1]
    print(num)
    
    #a = GetData("Circle", input_features)
