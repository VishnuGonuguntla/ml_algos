import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as anim

class LinearReg:
    def __init__ (self, eta, n, data):
        self.parameters = {}
        self.n = n
        self.eta = eta
        self.data = data
        self.process()
    def forward(self):
        return self.data["x"]*self.parameters["slope"] + self.parameters["intercept"]
    def backward(self, input):
        diff = input - self.data["y"]
        d_dm = 2*np.mean(self.data["x"]*diff)
        d_dc = 2*np.mean(diff)
        return d_dm, d_dc
    def cost_fn(self, input):
        return np.mean((input-self.data["y"])**2)
    def update_parameters(self,d_slope, d_intercept):
        self.parameters["slope"] -= self.eta*d_slope
        self.parameters["intercept"] -= self.eta*d_intercept
    def epoc(self,j):
        result = self.forward()
        cost = self.cost_fn(result)
        slope_update, intercept_update = self.backward(result)
        self.update_parameters(slope_update,intercept_update)
        self.plot(self.data["x"], self.data["x"]*self.parameters["slope"]-self.parameters["intercept"],"final_data")
        cost_updated = self.cost_fn(self.data["x"]*self.parameters["slope"]+self.parameters["intercept"])
        print(cost, cost_updated)
    def process(self):
        self.parameters["slope"] = -1*random.random()
        self.parameters["intercept"] = -1*random.random()
        slope = self.parameters["slope"]; intercept = self.parameters["intercept"]
        print(self.parameters["slope"], self.parameters["intercept"])
        fig = plt.figure(figsize=(10,6))
        plt.scatter(self.data["x"], self.data["y"])
        ani = anim.FuncAnimation(fig, self.epoc,frames=10, repeat=False, interval=100)
        plt.show()
        
        # To initialize the slope and intercept to initial values.
        # self.parameters["intercept"] = intercept; self.parameters["slope"] = slope
        # ani.save("animation.gif", writer="ffmpeg")
        
    def plot(self,x,y, data_label):
        plt.plot(x,y, label = data_label)