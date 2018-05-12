import numpy as np

class Neural_Network(object):
    def __init__(self):

        # Parameters
        self.inputSize = 784
        self.hiddenSize = 30
        self.outputSize = 10

        # Initial weights
        self.w1 = np.random.randn(self.inputSize,self.hiddenSize)
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize)

    def propagateForward(self, X):

        # Calculate net and out of the hidden layer
        self.netH = np.dot(X, self.w1)
        self.outH = sigmoid(self.netH, False)

        # Use outputes from hidden layer to calculate net and out of OUT
        self.netO = np.dot(self.outH, self.w2)
        self.outH = sigmoid(self.netO, False)

    def propagateBack(self, X):
        print("Go Back")

    def calculateError(self, X)
        return

    def sigmoid(self, x,deriv):
        if deriv:
            return x*(1-x)
        return 1/(1+np.exp(-x))


inputLayer = np.loadtxt("TrainDigitX.csv", delimiter=",")
outputLayer = np.loadtxt("TrainDigitY.csv")

print(sigmoid(5,False))
