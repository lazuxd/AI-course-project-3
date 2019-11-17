import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib import animation
import numpy as np
import sys
from sys import maxsize as MAX_INT
from typing import List, Tuple

def plotData(age, weight, height, n_age, n_weight, n_height):
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].scatter(age, np.linspace(0, 0, len(age)), c='#0961ed')
    ax[0].scatter(weight, np.linspace(1, 1, len(weight)), c='#18ed09')
    ax[0].scatter(height, np.linspace(2, 2, len(height)), c='#f55905')

    ax[1].scatter(n_age, np.linspace(0, 0, len(n_age)), c='#0961ed')
    ax[1].scatter(n_weight, np.linspace(1, 1, len(n_weight)), c='#18ed09')
    ax[1].scatter(n_height, np.linspace(2, 2, len(n_height)), c='#f55905')

    plt.show()

class LinearModel:
    def __init__(self, alpha = 0.001):
        self.reset(alpha)
    
    def __h(self, a, w):
        return np.dot(self.betas, [1, a, w])
    
    def reset(self, alpha = 0.001):
        self.betas = [0, 0, 0]
        self.alpha = alpha
        self.threshold = 0.0001
        self.iterationWhenThresholdReached = MAX_INT
        self.gradientMagnitude = MAX_INT
        self.iterationsSoFar = 0
    
    def gdIteration(self, a, w, h, n = None, data_matrix = None, has_matrix_data = False):
        if n == None:
            n = len(a)
        ph = list(map(lambda ai, wi: self.__h(ai, wi), a, w))
        diff = np.subtract(ph, h)
        if has_matrix_data == False:
            data_matrix = np.stack((np.linspace(1,1,n), a, w))
        gradient = (1/n)*np.dot(data_matrix, diff)
        self.betas = np.subtract(self.betas, self.alpha*gradient)

        self.gradientMagnitude = np.sqrt(np.dot(gradient, gradient))
        self.iterationsSoFar = self.iterationsSoFar + 1
        if self.iterationWhenThresholdReached == MAX_INT and self.gradientMagnitude <= self.threshold:
            self.iterationWhenThresholdReached = self.iterationsSoFar
    
    def learn(self, a, w, h, iterations = 100):
        n = len(a)
        if n != len(w) or n != len(h):
            raise Exception('All vectors should be the same size.')
        data_matrix = np.stack((np.linspace(1,1,n), a, w))

        for i in range(iterations):
            self.gdIteration(a, w, h, n, data_matrix, has_matrix_data=True)
    
    def predict(self, age, weight):
        return self.__h(age, weight)
    
    def getBetas(self):
        return self.betas
    
    def getIterationWhenThresholdReached(self):
        return self.iterationWhenThresholdReached
    
    def getGradientMagnitude(self):
        return self.gradientMagnitude

if __name__ == '__main__':
    inputFilename: str = sys.argv[1]
    outputFilename: str = sys.argv[2]
    if not inputFilename or not outputFilename:
        print('You forgot the INPUT and OUTPUT files!')
        exit()
    
    age: List[float] = []
    weight: List[float] = []
    height: List[float] = []

    with open(inputFilename, "r") as inputFile:
        for line in inputFile:
            [a, w, h] = list(map(lambda x: float(x), line.split(',')))
            age.append(a)
            weight.append(w)
            height.append(h)

    mean_age = sum(age) / len(age)
    mean_weight = sum(weight) / len(weight)
    mean_height = sum(height) / len(height)

    std_dev_age = np.std(age)
    std_dev_weight = np.std(weight)
    std_dev_height = np.std(height)

    n_age = list(map(lambda a: (a-mean_age)/std_dev_age, age))
    n_weight = list(map(lambda w: (w-mean_weight)/std_dev_weight, weight))
    n_height = list(map(lambda h: (h-mean_height)/std_dev_height, height))

    # plotData(age, weight, height, n_age, n_weight, n_height)

    xs = np.linspace(2, 8.46, 100)
    ys = np.linspace(10.21, 40.95, 100)
    xs, ys = np.meshgrid(xs, ys)

    fig = plt.figure() #type: Figure
    ax = fig.add_subplot(111, projection='3d') #type: Axes3D

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.0049]
    iterations = [100, 100, 100, 100, 100, 100, 100, 100, 100, 15000]

    def findOptimal(alpha, iterations):
        lm = LinearModel(alpha)
        lm.learn(age, weight, height, iterations)
        betas = lm.getBetas()
        print(f'Training with alpha={alpha}')
        print('Iteration when threshold reached: ', lm.getIterationWhenThresholdReached())
        print('Gradient magnitude: ', lm.getGradientMagnitude())
        print('Betas: ', lm.getBetas())
        ax.clear()
        ax.set_title(f'alpha = {alpha} ; iterations = {iterations}')
        ax.set_xlabel('Age (Years)')
        ax.set_ylabel('Weight (Kilograms)')
        ax.set_zlabel('Height (Meters)')
        ax.scatter(age, weight, height)
        zs = betas[0] + betas[1]*xs + betas[2]*ys
        ax.plot_surface(xs, ys, zs, cmap = plt.get_cmap('autumn'), linewidth=0)
        plt.show()

    def writeResultsToFile():
        with open(outputFilename, "w") as outputFile:
            for i in range(len(alphas)):
                lm = LinearModel(alphas[i])
                lm.learn(age, weight, height, iterations[i])
                lineData = [alphas[i], iterations[i]]
                lineData.extend(lm.getBetas())
                outputFile.write(','.join(map(lambda x: str(x), lineData))+'\n')

    def showAnimation():
        idx = [0]
        lm = LinearModel(alphas[0])
        def animate(i):
            j = idx[0]
            if iterations[j] == 0:
                j = j+1
                idx[0] = j
                print(f'#{j}: Training with alpha={alphas[j-1]}')
                print('Iteration when threshold reached: ', lm.getIterationWhenThresholdReached())
                print('Gradient magnitude: ', lm.getGradientMagnitude())
                print('Betas: ', lm.getBetas())
                print('-------------------------------------------------------------------------')
                if j >= len(alphas):
                    exit()
                lm.reset(alphas[j])
            
            iterations[j] = iterations[j]-1
            lm.gdIteration(age, weight, height)
            betas = lm.getBetas()
            zs = betas[0] + betas[1]*xs + betas[2]*ys

            ax.clear()
            ax.set_title(f'alpha = {alphas[j]} ; it. left: {iterations[j]}')
            ax.set_xlabel('Age (Years)')
            ax.set_ylabel('Weight (Kilograms)')
            ax.set_zlabel('Height (Meters)')
            ax.scatter(age, weight, height)
            ax.plot_surface(xs, ys, zs, cmap = plt.get_cmap('autumn'), linewidth=0)

        anim = animation.FuncAnimation(fig, animate, 200)

        plt.show()

    # showAnimation()
    # writeResultsToFile()
    findOptimal(0.0049, 15000)

