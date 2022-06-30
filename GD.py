import numpy as np
import matplotlib.pyplot as plt
#Dependencies

def func(x): # Function to approximate
    return np.sin(3*x)
x = np.arange(-1, 1, 0.01)
y = [func(i) for i in x]

def mse(y, y_hat): # Loss function
    return (y - y_hat)**2

def d_mse(y, y_hat): # Graadient of the loss function
    return -2 * (y - y_hat)

class Model(): # Function approximator
    def __init__(self, x, y, lr=0.01, degree=2, loss=mse, d_loss=d_mse):
        self.obs = x
        self.y = y
        self.lr = lr
        self.coefficients = np.random.rand(degree + 1) #initializing the parameters to optimize
        self.loss, self.d_loss = loss, d_loss
        self.errors = []
    
    def train(self): #Iteratively optimizing the parameters using gradient descent
        l = len(self.coefficients)
        grads = np.zeros(l)
        for i in range(len(self.obs)):
            y_hat = self.evaluate(self.obs[i])
            for t in range(l):
                grads[t] = self.d_loss(self.y[i], y_hat) * self.obs[i] ** (l-t-1) * self.lr
            self.coefficients -= grads
        self.errors.append(np.mean([self.loss(self.y[i], self.evaluate(self.obs[i])) for i in range(len(self.obs))]))
        
    def evaluate(self, x): #outputs the model's prediction given an input x
        l = len(self.coefficients)
        s = 0
        for i in range(l):
            s += self.coefficients[i] * x ** (l - i - 1)
        return s

model = Model(x, y, degree=10, lr=0.1)#Instantiating the model

for i in range(500): #Training the model
    model.train()

#plotting the model's approximation along with the actual function
plt.figure(figsize=(20,10))
plt.plot(x, y)
plt.plot(x, [model.evaluate(i) for i in x])

#Plotting the model's error as the model trained
plt.figure(figsize=(20,10))
plt.plot(model.errors)
