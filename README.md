# Gradient-Descent
Optimization of polynomial coefficients using gradient descent for approximating functions.

Programmed here is a function approximator (Model) whos primary objective is to approximatea certain target function (func) by using Gradient Descent.

The function approximator built here seems to be very effective at approximating functions. After training with hundreds of thousands of samples of data, the approximator (depending on the target function) achieves very low errors. Gradient descent easily rivals other approximation techniques like linear regression. However given a highly periodic target function like sin(10x), the approximator struggles to find ideal parameters, for functions like those, a Taylor Series approximation may prove to be more accurate than what is built here.
