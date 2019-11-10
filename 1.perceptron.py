from sys import argv
from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy

class Perceptron:
    def __init__(self):
        self.w1 = 0.0
        self.w2 = 0.0
        self.b = 0.0

    def __g(self, x: float, y: float) -> float:
        return self.w1*x + self.w2*y + self.b
    
    def __f(self, x: float, y: float) -> int:
        g = self.__g(x, y)
        if g > 0:
            return 1
        else:
            return -1

    def update(self, x: float, y: float, label: int) -> bool:
        f = self.__f(x, y)
        if f*label <= 0:
            self.w1 = self.w1 + label*x
            self.w2 = self.w2 + label*y
            self.b = self.b + label
            return True
        else:
            return False
    
    def getWeights(self) -> Tuple[float]:
        return (self.w1, self.w2, self.b)


if __name__ == '__main__':
    inputFilename: str = argv[1]
    outputFilename: str = argv[2]
    if not inputFilename or not outputFilename:
        print('You forgot the INPUT and OUTPUT files!')
        exit()

    x: List[float] = []
    y: List[float] = []
    label: List[int] = []
    colors: List[str] = []
    perceptron = Perceptron()
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlim((-1, 16))
    plt.ylim((-25, 25))
    plt.autoscale(False)
    t = numpy.linspace(0, 15, 1000)
    done = [False]
    outputFile = open(outputFilename, "w")

    def plotPerceptron(w1, w2, b):
        if w2 == 0.0:
            w2 = 0.001
        ax.plot(t, -(w1*t+b)/w2, color = 'black')

    def animate(_):
        if done[0]:
            return False
        
        updated = False
        for i in range(len(x)):
            if perceptron.update(x[i], y[i], label[i]):
                updated = True

        if not updated:
            done[0] = True
            return False

        (w1, w2, b) = perceptron.getWeights()
        outputFile.write(','.join(map(lambda x: str(x), (w1, w2, b)))+'\n')

        ax.clear()
        plt.xlim((-1, 16))
        plt.ylim((-25, 25))
        plt.autoscale(False)
        plt.scatter(x, y, c=colors)
        plotPerceptron(w1, w2, b)

    with open(inputFilename, "r") as inputFile:
        for line in inputFile:
            (x_i, y_i, label_i) = tuple(map(lambda x: int(x), line.split(',')))
            x.append(x_i)
            y.append(y_i)
            label.append(label_i)
            colors.append('#0a8af2' if label_i == 1 else '#f20e0a')
    
    plt.scatter(x, y, c=colors)
    (w1, w2, b) = perceptron.getWeights()
    plotPerceptron(w1, w2, b)
    anim = animation.FuncAnimation(fig, animate, interval=400)
    plt.show()

    outputFile.close()
