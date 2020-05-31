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
    data = pg.generate_data(data, DATA_NOISE)
    data = pg.split_data(data,
                         training_size=TRAINING_DATA_RATIO)
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
    def __init__(self, DataName, evaluate=True):
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
            x = np.linspace(x1_min, x1_max, 100)
            y = np.linspace(x2_max, x2_min, 100)
            X,Y = np.meshgrid(x, y)
            e_data = np.stack((X,Y), -1)
            self.X, self.Y = X, Y
            self.evaluate_data = np.reshape(e_data, (-1,2))
        else:
            self.evaluate_data = None

    
        self.fig, self.ax = plot_setting(x1_min, x1_max, x2_min, x2_max)
        self.mpl_draw()
        
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
        sc = self.ax.pcolormesh(self.X,
                                self.Y,
                                data,
                                vmin=0,
                                vmax=1,
                                cmap="coolwarm_r")
        cn = self.ax.contour(self.X,
                             self.Y,
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
    a = GetData("Circle")
