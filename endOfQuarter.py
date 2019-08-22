import os
import numpy as np


# from itertools import combinations
# import pprint
# import time
# from numpy.linalg import inv
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')

port = os.getenv('PORT', '8080')
host = os.getenv('IP', '0.0.0.0')


# Not used
def wRandom(n, last):
    # Make w a random vector, length n
    # Last is a zero vector, length n
    # There's a one-in-a-gazillion chance of w = [0,0,0...]
    # Handle that with a loop
    while True:
        w = np.random.randint(-2, 2, size=n)
        if not np.array_equal(w, last):
            return w


# ==============================================================
# Gradient Descent Sochastic and Batch
# ==============================================================


def exampleData():
    # x1 x2 x3 y
    # cloudy=0, reining=1, sunny=2
    X = np.array([
        [1, 0.45, 3.25],
        [1, -1.08, 2.2],
        [1, .2, 1.18],
        [1, -1.18, .98],
        [1, -2.49, 3.59]
    ])
    Y = np.array([1, -1, -1, 1, 1])
    return (X, Y)


def sigmoid(fx):
    return 1 / (1 + np.exp(fx * -1))


def gradDescent_sochastic(X, Y, w, eta):
    halt = 1
    for h in range(0, halt):
        g = np.zeros(len(w))
        print('g:', g)
        for i in range(0, len(Y)):
            # This outputs current w to match slide set 7 example
            print('w = ', w)
            s = np.dot(w, X[i])   # wtXi
            s = np.dot(Y[i], s)   # yi*wtxi
            s = sigmoid(s)        # sig(yiwtxi)
            s = 1 - s                 # (1-pi)
            s *= Y[i]               # (1-pi)yi
            s = np.dot(X[i], s)   # (1-pi)yixi
            s = np.dot(eta, s)    # eta*(1-pi)yixi
            w = np.add(w, s)      # w = w+eta*(1-pi)yixi
    return w


def gradDescent_batch(X, Y, w, eta):
    halt = 1
    for h in range(0, halt):
        g = np.zeros(len(w))
        print('g:', g)
        for i in range(0, len(Y)):
            print('g = ', g)
            s = np.dot(w, X[i])   # wtXi
            s = np.dot(Y[i], s)   # yi*wtxi
            s = sigmoid(s)        # sig(yiwtxi)
            s = 1 - s             # (1-pi)
            s *= Y[i]             # (1-pi)yi
            s = np.dot(X[i], s)   # (1-pi)yixi
            s = np.dot(s, -1)     # -(1-pi)yixi
            g = np.add(g, s)
        g = np.dot(g, 1 / len(Y))   # 1/m(-(1-pi)yixi)
        g = np.dot(eta, g)
        w = np.subtract(w, g)
    return w


def test_exampleData():
    (X, Y) = exampleData()
    print('X')
    print(X)
    print('Y')
    print(Y)
    w = np.array([-1, 1, 1])
    eta = 0.1
    w = gradDescent_batch(X, Y, w, eta)
    print('final w')
    print(w)

# ==============================================================
# Gradient Descent Batch with Added Regularizer
# Cross Validation with different lambdas
# ==============================================================


# For no regularization
def reg0(lam, w, i):
    return np.zeros(len(w))


def reg(lam, m, w):
    w1 = np.insert(w[1:], 0, 0)
    return np.dot(w1, 2 * lam / m)


def gradDescent(X, Y, w, eta, lam):
    hardHalt = 100
    condHalt = 1e-10
    minL = 10000000
    for h in range(0, hardHalt):
        # w only changes at end of batch, so just add it to g now
        # Also, I would argue that it's not necessary to divide by m, since
        # the batch calculates w over m
        g = reg(lam, 1, w)  # regularization
        for i in range(0, len(Y)):
            s = np.dot(w, X[i])   # wtXi
            s = np.dot(Y[i], s)   # yi*wtxi
            s = sigmoid(s)        # sig(yiwtxi)
            s = (1 - s)               # (1-pi)
            s *= Y[i]               # (1-pi)yi
            s = np.dot(X[i], s)   # (1-pi)yixi
            s = np.dot(s, -1)     # -(1-pi)yixi
            g = np.add(g, s)
        g = np.dot(g, 1 / len(Y))  # 1/m(-(1-pi)yixi)
        g = np.dot(eta, g)
        w = np.subtract(w, g)
        lensq = np.dot(g, g)
        if h and lensq < condHalt:
            print('condHalt @ h = ', h, minL)
            return w
        if lensq < minL:
            minL = lensq
        if h % 10000 == 0:
            print('minL', lam, h, minL)
    # print('no halt, minL = ', minL)
    return w


# load the data
def loadsparsedata(fn):
    fp = open(fn, "r")
    lines = fp.readlines()
    maxf = 0
    for line in lines:
        for i in line.split()[1::2]:
            maxf = max(maxf, int(i))

    X = np.zeros((len(lines), maxf))
    Y = np.zeros((len(lines)))

    for i, line in enumerate(lines):
        values = line.split()
        Y[i] = int(values[0])
        for j, v in zip(values[1::2], values[2::2]):
            X[i, int(j) - 1] = int(v)

    return X, Y


def getXY(fn):  # load from file and fix
    (X, Y) = loadsparsedata(fn)
    # add column of 1s as zeroth feature
    X = np.column_stack((np.ones(X.shape[0]), X))
    Y[Y == 0] = -1
    return X, Y


def learnlogreg(X, Y, lam):
    (m, n) = X.shape
    w = np.zeros((n))
    eta = np.sqrt(2.5 / lam)
    return gradDescent(X, Y, w, eta, lam)


def linearerror(X, Y, w):
    # returns error *rate* for linear classifier with coefficients w
    m = Y.shape[0]
    predy = X.dot(w)
    err = (Y[predy >= 0] < 0.5).sum() + (Y[predy < 0] >= 0.5).sum()
    return err / m


def crossval(trainFile, lambdas, div):
    if(div and div - 1):        # assert non-zero denominator
        denom = div * (div - 1)   # for avg over number of tests
    else:
        return None

    (X, Y) = getXY(trainFile)
    Xcross = np.split(X, div)  # returns list
    Ycross = np.split(Y, div)
    errs = np.zeros(len(lambdas))
    bestW = 0
    bestLam = 0
    minErr = 1000000
    h = 0

    for lam in lambdas:
        for V, VY in zip(Xcross, Ycross):  # the splits of X, Y
            w = learnlogreg(V, VY, lam)
            curErr = 0
            for T, TY in zip(Xcross, Ycross):  # splits of X, Y that are not V
                if V is not T:
                    curErr += linearerror(T, TY, w)
            if(curErr < minErr):
                minErr = curErr
                bestW = w
                bestLam = lam
            errs[h] += curErr
        errs[h] /= denom
        h += 1
    return (bestLam, bestW, errs)


def trainTest(trainFile, testFile, lambdas):
    (trainX, trainY) = getXY("spamtrain.txt")
    (testX, testY) = getXY("spamtest.txt")
    errs = np.zeros(len(lambdas))
    bestW = 0
    bestLam = 0
    minErr = 1000000
    h = 0
    for lam in lambdas:
        w = learnlogreg(trainX, trainY, lam)
        errs[h] = linearerror(testX, testY, w)
        if(errs[h] < minErr):
            minErr = errs[h]
            bestW = w
            bestLam = lam
        h += 1
    return (bestLam, bestW, errs)


# ==============================================================
# Gradient Descent Batch with Added Regularizer
# Cross Validation with different lambdas
# ==============================================================


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork2:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.array(
            [
                [0.85857033, 0.89037339, 0.69891232, 0.83096068],
                [0.27753788, 0.20586593, 0.30851418, 0.3868346],
                [0.52653618, 0.5921138, 0.66015643, 0.69332094]
            ]
        )
        self.weights2 = np.array(
            [[0.9811953], [0.25472213], [0.11068546], [0.35546831]])
        self.y = y
        self.out = np.zeros(self.y.shape)

    def nneval(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.out = sigmoid(np.dot(self.layer1, self.weights2))

    def propBack(self, i):
        # application of the chain rule to find derivative of
        # the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(
            self.layer1.T,
            (2 * (self.y - self.out) * sigmoid_derivative(self.out))
        )
        d_weights1 = np.dot(
            self.input.T,
            (
                np.dot(
                    2 * (self.y - self.out) * sigmoid_derivative(self.out),
                    self.weights2.T
                ) * sigmoid_derivative(self.layer1)
            )
        )

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


def deriv_sig(fx):
    return fx * (1 - fx)


class NeuNet_simple:
    def __init__(self, X, Y, numHid):
        self.weights = []
        self.ders = []
        # m = X.shape[0]
        # n = X.shape[1]
        self.weights.append(
            np.array([[0.85857033, 0.89037339, 0.69891232, 0.83096068],
                      [0.27753788, 0.20586593, 0.30851418, 0.3868346],
                      [0.52653618, 0.5921138, 0.66015643, 0.69332094]])
        )
        self.weights.append(
            np.array([[0.9811953], [0.25472213], [0.11068546], [0.35546831]])
        )
        # for i in range(numHid+1):
        #     self.weights.append(np.random.rand(n, m))
        #     print("wi shape",i, self.weights[i].shape)
        #     n = m
        #     m = 1

    def nneval(self, X):
        self.layers = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.layers[i], self.weights[i])
            # print("z shape",i, z.shape)
            self.layers.append(sigmoid(z))
            # print("layeri shape", i, self.layers[i].shape)
            self.ders.append(np.zeros(self.weights[i].shape))

        self.out = self.layers[-1]

    def trainneuralnet(self, Y):
        derLoss = 2 * (Y - self.out)
        derSig = deriv_sig(self.out)
        self.layers[-1] = derSig * derLoss
        for i in range(len(self.layers) - 1, 0, -1):
            # print("z shape", i, self.layers[i].shape)
            self.ders[i - 1] = np.dot(self.layers[i - 1].T, self.layers[i])
            # print("deri shape", i - 1, self.ders[i - 1].shape)
            a = np.dot(self.layers[i], self.weights[i - 1].T)
            # print("a shape", i, a.shape)
            z = a * deriv_sig(self.layers[i - 1])
            # print("z shape", i - 1, z.shape)
            self.weights[i - 1] = np.add(self.weights[i - 1], self.ders[i - 1])
            self.layers[i - 1] = z


def testNNSimple():
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork2(X, Y)

    for i in range(1500):
        nn.nneval()
        nn.propBack(i)

    print(nn.out)
    print(nn.weights1)
    print(nn.weights2)

    print("===========")
    nn1 = NeuNet_simple(X, Y, 1)

    for i in range(1500):
        nn1.nneval(X)
        nn1.trainneuralnet(Y)

    print(nn1.out)
    print(nn1.weights[0])
    print(nn1.weights[1])


def nonLin(fn):
    return sigmoid(fn)
    # return fn


def fix(arr):
    return arr.reshape(1, -1) if len(arr.shape) < 2 else arr


class NeuNet_example:
    # def __init__(self, X):
    #     self.layers = [X]

    def nneval(self, X, w):
        self.layers = [X]
        self.weights = w
        self.ders = []
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], self.layers[i])
            a = nonLin(z)
            if i < (len(self.weights) - 1):
                self.layers.append(np.insert(a, 0, 1))
            else:
                self.layers.append(a)
            self.ders.append(np.zeros(self.weights[i].shape))
        self.out = self.layers[-1]

    def trainneuralnet(self, Y):
        self.layers[-1] = (self.out - Y)
        self.layers[-1] = np.insert(fix(self.layers[-1]), 0, 1)

        for i in range(len(self.layers) - 1, 0, -1):
            print()
            print("loop =", i)
            self.layers[i] = np.delete(self.layers[i], 0, 0)
            self.ders[i - 1] = np.dot(fix(self.layers[i]),
                                      fix(self.layers[i - 1]))
            print("ders i", i - 1,
                  fix(self.ders[i - 1]), fix(self.ders[i - 1]).shape)
            a = np.dot(fix(self.weights[i - 1]).T, fix(self.layers[i]))
            print("new a", i - 1, a, a.shape)
            z = a * deriv_sig(fix(self.layers[i - 1]).T)
            print("z", i - 1, z, z.shape)
            self.layers[i - 1] = z
            self.weights[i - 1] = np.add(fix(self.weights[i - 1]),
                                         fix(self.ders[i - 1]))


def nnExampleTest():
    X = np.array([1, 2, -1])
    Y = np.array([1])
    weights = [
        np.array([[-1, 2, 2], [0, -1, 3], [1, 5, 0]]),
        np.array([[0, 0, 1, 2], [-2, 1, -1, 1]]),
        np.array([4, -5, 2])
    ]
    print("X:", X.shape)
    print("Y:", Y.shape)
    for i in range(len(weights)):
        print("w:", weights[i].shape)
    # X = np.insert(X, 0, 5)
    # print(X)
    # X = np.delete(X, 0, 0)
    # print(X)
    nn = NeuNet_example()
    nn.nneval(X, weights)
    print()
    print(nn.out)
    print("trainneuralnet")
    nn.trainneuralnet(Y)
    print(nn.out)
    # print(nn.weights[0])
    # print(nn.weights[1])


def gt(a, b):
    out = 0
    if(a.shape != b.shape):
        print("gt fail")
        return 0
    for ai, bi in zip(a, b):
        for aj, bj in zip(ai, bi):
            out += 1 if abs(aj) > abs(bj) else -1
    return out > 0

# ==============================================================
# Implementation of neural net
# ==============================================================

# def sigmoid(fx):
#     return 1/(1 + np.exp(fx*-1))

# def deriv_sig(fx):
#     return fx * (1 - fx)

# def nonLin(fn):
#     return sigmoid(fn)


def nneval(X, W1, W2):
    W = [W1, W2]
    layers = [X]                          # X is first layer
    for i in range(len(W)):
        layers[i] = np.append(               # Add column of ones
            layers[i],
            np.ones((len(layers[i]), 1)),
            axis=1
        )
        z = np.dot(layers[i], W[i].T)
        layers.append(nonLin(z))

    return layers[-1]


def fx(arr):
    return arr[:, np.newaxis] if len(arr.shape) < 2 else arr


OF_ERR = False


class NeuNet:
    def __init__(self, X, Y, W, lam):
        self.X = X
        self.Y = Y
        self.weights = W
        self.lam = lam
        # init gradient
        self.grads = []
        for i in range(len(self.weights)):
            self.grads.append(np.zeros(self.weights[i].shape))
        # init for eta calculations
        self.eg2 = 1
        self.eta = 1e-10
        # For Halt
        self.minSum = 1e6
        self.curSum = 0
        self.last = 1

    def forward(self):
        self.layers = [self.X]
        for i in range(len(self.weights)):
            self.layers[i] = np.append(
                self.layers[i],
                np.ones((len(self.layers[i]), 1)),
                axis=1
            )
            z = np.dot(self.layers[i], self.weights[i].T)
            a = nonLin(z)
            self.layers.append(a)
            if OF_ERR:
                return
        self.out = self.layers[-1]

    def loss(self):
        derLoss = 2 * (self.Y - self.out)
        derSig = deriv_sig(self.out)
        return derSig * derLoss

    def back(self):
        self.layers[-1] = self.loss()
        self.layers[-1] = np.append(self.layers[-1],
                                    np.ones((len(self.layers[-1]), 1)), axis=1)
        for i in range(len(self.layers) - 1, 0, -1):

            self.layers[i] = fx(self.layers[i])[:, :-1]
            self.grads[i - 1] = np.dot(self.layers[i].T, self.layers[i - 1])
            a = np.dot(fx(self.layers[i]), fx(self.weights[i - 1]))
            z = a * deriv_sig(fx(self.layers[i - 1]))
            self.layers[i - 1] = z
            if OF_ERR:
                return

    def update(self):
        # Loop handles four tasks:
        # 1. sum of gradients squared, for eta calc
        # 2. Update the weights
        # 3. Set regularization/clear the gradients
        # 4. Set halt on low gradient
        self.curSum = 0
        for i in range(len(self.weights)):
            # 1. sum of gradients squared, for eta calc
            g = self.grads[i].reshape(self.grads[i].size, 1)
            for j in range(len(g)):
                self.curSum = g[i, 0] * g[i, 0]
            # 2. Update the weights
            self.weights[i] -= self.eta * self.grads[i]
            # 3. Set regularization/clear the gradients
            # 2*self.lam*self.weights[i]
            self.grads[i] = np.zeros(self.weights[i].shape)
            if OF_ERR:
                return
        # 4. Set halt on low gradient
        if self.curSum < self.minSum:
            self.minSum = self.curSum
        # Recalc eta
        self.eg2 = 0.9 * self.eg2 + 0.1 * self.curSum
        self.eta = 0.001 / np.sqrt(1e-10 + self.eg2)
        # print("on update", self.curSum, self.eg2, self.eta)

    def done(self):
        # print("curSum:",self.curSum, "\t\t", self.eta)
        if(abs(self.last - self.curSum) < 10e-6):
            return True
        self.last = self.curSum
        return False


def err_handler(type, flag):
    # print("Err Handler Called yeah!")
    OF_ERR = True


saved_handler = np.seterrcall(err_handler)
save_err = np.seterr(all='call')


def trainneuralnet(X, Y, nhid, lam):
    print("trainneuralnet, nhid, lam", nhid, lam)
    (m, n) = X.shape
    W = [
        np.random.randn(nhid, n + 1) / n,
        np.random.randn(1, nhid + 1) / nhid
    ]
    # print(W[0], W[0].shape)
    nn = NeuNet(X, Y, W, lam)
    # print("running...")
    for i in range(100):  # 100000
        nn.forward()
        # if OF_ERR:
        #     print("Err forward!")
        #     return False
        nn.back()
        # if OF_ERR:
        #     print("Err back!")
        #     return False
        nn.update()
        # if OF_ERR:
        #     print("Err Update!")
        #     return False
        # if(i%5000 == 0):
        #     print("Running...", i)
        # if(i%1000 == 0 and nn.done()):
        #     print("Done on loop", i)
        #     return tuple(nn.weights)
    return tuple(nn.weights)


def restartOnErr(X, Y, nhid, lam):
    W = False
    while not W:
        OF_ERR = False
        W = trainneuralnet(X, Y, nhid, lam)
    return W


def prob1():
    # args
    nhid = 2
    lam = .1

    X = np.random.rand(5, 4)
    Y = np.random.rand(5, 1)
    (W1, W2) = restartOnErr(X, Y, nhid, lam)
    print(" = Done============================================")
    print()
    print("Shapes", W1.shape, W2.shape)
    print(W1)
    print()
    print(W2)


def testNneval():
    # args
    nhid = 2
    # lam = .01

    X = np.random.rand(5, 4)
    # Y = np.random.rand(5, 1)
    (m, n) = X.shape
    W = [
        np.random.randn(nhid, n + 1) / n,
        np.random.randn(1, nhid + 1) / nhid
    ]
    yy = nneval(X, W[0], W[1])
    print("yy shape", yy.shape)
    print("yy", yy)

# ==============================================================
# Utility functions, some related to cluster class below
# ==============================================================


def choose(n, k):
    # Choose function implements binomial coefficient formula
    if(k < 0 or (n - k) < 0):       # junk paramaters
        print('Error {a:8.2f} choose {b:8.2f}'.format(a=n, b=k))
        return 0
    if(k == 0):                     # base case
        return 1
    newn, newk = 1, 1
    for i in range(n, (n - k), -1):  # n!/(n-k)!
        newn = newn * i
    for i in range(k, 0, -1):       # k!
        newk = newk * i
    return newn / newk


def nAssocRule(m):
    outerSum = 0
    for k in range(1, m):
        str1 = ""
        mck = choose(m, k)
        innerSum = 0
        for j in range(1, m - k + 1):
            innerSum = choose(m - k, j)
            print(
                '{:d} choose {:d} * {:d} choose {:d}'.format(m, k, (m - k), j)
            )
            str1 = '*'
        outerSum = mck * innerSum
        # print("m, outer", m, outerSum)
        print(str1)
    return outerSum


def nAssocRule2(d):
    return pow(3, d) - pow(2, d + 1) + 1


def settostr(s, names=None):  # prints out a setnames is a list
    if names is None:
        elems = [str(e) for e in s]
    else:
        elems = [names[e] for e in s]
    return "{" + (", ".join(elems)) + "}"


def inList(needle, haystack):
    for i in range(len(haystack)):
        if haystack[i] == needle:
            return 1
    return 0


def isSubset(needle, haystack):  # compare two 1-dimensional lists
    if len(needle) > len(haystack):
        return 0
    for i in range(len(needle)):
        found = False
        for j in range(len(haystack)):
            if needle[i] == haystack[j]:
                found = True
                break
        if not found:
            return 0
    return 1


def common(A, B):  # compare two 1-dimensional lists
    if len(A) < len(B):
        return isSubset(A, B)
    return isSubset(B, A)


def makeSet(myset):  # powerset
    if not myset:
        return [set()]
    r = []
    for y in myset:
        sy = set((y,))
        for x in makeSet(myset - sy):
            if x not in r:
                r.extend([x, x | sy])
    return r


def sameList(A, B):  # assumes lists are sorted
    if(len(A) != len(B)):
        return 0
    for i in range(len(A)):
        if A[i] != B[i]:
            return 0
    return 1

# ==============================================================
# Class for clustering
# Exponential runtime :-(
# ==============================================================


class assocRules:
    # Class Members:
    # nItems, mSup, mConf           //values
    # data, names                   //lists
    # rules, supCounts              //dicts
    def __init__(self, filename):
        self.rules = {}
        with open(filename) as f:
            self.nItems = int(f.readline())
            self.names = [f.readline().strip() for i in range(self.nItems)]
            nrows = int(f.readline())
            self.data = [[int(s) for s in f.readline().split()]
                         for i in range(nrows)]

            f.close()
            print("nItems", self.nItems)
            print("nItems", len(self.names))
            print("nrows", nrows)
            print("nrows", len(self.data))

            self.names = None

    def countSingles(self):  # item must be a list
        # makes a hash from item contents, then calculates count
        # if called again on same item, returns stored count
        self.singles = [0] * self.nItems
        for line in self.data:
            for item in line:
                self.singles[item] = 1

    def pruneSingles(self):
        # Before making rules, remove single items with low support
        for i in range(len(self.data)):
            cur = []
            for j in range(len(self.data[i])):
                if self.singles[self.data[i][j]] >= self.mSup:
                    cur.append(self.data[i][j])
            self.data[i] = cur

    def pruneSames(self):
        sames = [0] * len(self.data)
        out = []
        for i in range(0, len(self.data)):
            for j in range(i + 1, len(self.data)):
                sames[i] = sameList(self.data[i], self.data[j])
        for i in range(0, len(self.data)):
            if(sames[i] == 0):
                out.append(self.data[i])
        self.data = out

    def countOccurs(self, item):  # item must be a list
        # makes a hash from item contents, then calculates count
        # if called again on same item, returns stored count
        iHash = ','.join([str(x) for x in item])
        if(iHash in self.supCounts):          # old item
            return self.supCounts[iHash]
        # if (len(self.supCounts)%1000) == 0:
        #     print("supCounts size", len(self.supCounts))
        self.supCounts[iHash] = 0               # new item: calculate
        for i in range(len(self.data)):
            if isSubset(item, self.data[i]):
                self.supCounts[iHash] = 1
        return self.supCounts[iHash]

    def addRule(self, lhs, rhs):
        lhs.sort()
        rhs.sort()
        # Need a unique name
        iHash = ','.join([str(x) for x in lhs]) + '|' + \
            ','.join([str(x) for x in rhs])
        # Don't add more than once
        if(iHash in self.rules):
            return
        # Skip adding low support and low confidence rules
        sup = self.countOccurs(lhs + rhs)
        if(sup < self.mSup):
            # print("MSUP")
            return
        conf = sup / self.countOccurs(lhs)
        if(conf < self.mConf):
            # print("MCONF")
            return
        self.rules[iHash] = [lhs, rhs, sup / len(self.data), conf]

    def makeRules(self, line):
        n = len(line)
        pset = makeSet(set(line))
        pset = list([list(i) for i in pset if len(i)])
        for i in range(len(pset)):
            for j in range(len(pset)):
                if i != j and \
                        not (len(pset[i]) + len(pset[j]) > n) and \
                        not common(pset[i], pset[j]):
                    pass
                    # self.addRule(pset[i], pset[j])
        # print("rul len =",len(self.rules))

    def learnrules(self, mSup, mConf):
        # start_time = time.time()
        # using the count rather than probability
        self.mSup = mSup * len(self.data)
        self.mConf = mConf
        print("done loading")
        self.countSingles()
        print(self.singles)
        self.supCounts = {}
        print("early prune")
        self.pruneSingles()
        # self.dispData()
        # print("supCounts size", len(self.supCounts))
        # print("--- %s seconds ---" % (time.time() - start_time))
        # self.rules={}

        # for line in self.data:
        #     self.makeRules(line)
        # print("--- %s seconds ---" % (time.time() - start_time))

    def dispData(self):
        print('============Display Data==============================')
        c = 0
        for line in self.data:
            if len(line):
                c = 1
                print(settostr(line, self.names))
        print("Data len =", c)
        print('============End Data==================================')
        print()

    def writerules(self):
        # m = len(self.data)
        print('============Display Rules=============================')
        print("supCounts size", len(self.supCounts))
        print("Rules len =", len(self.rules))


def printruleset(fName, mSup, mConf):
    rs = assocRules(fName)
    rs.dispData()
    rs.learnrules(mSup, mConf)
    # rs.writerules()


def main():
    printruleset("groceries2.txt", .01, .5)
    # printruleset("toymovies.txt",.5,.5)


if __name__ == '__main__':
    main()

    # [0, -1,0],
    # [0, -1, 1],
    # [0,0, 1],
    # [1, -1,0],
    # [0, 1,0],
    # [0, 1, 1],
    # [1,0, 1],
    # [0, -1,0],
    # [1, -1,0],
    # [0,0,0]

    # [-1, -1, 1, -1, -1, 1, 1, -1, 1, -1]

    # print("before", X, X.shape)
    # X = fix(X)
    # print("after", X, X.shape)

    # print("before", weights[0], weights[0].shape)
    # weights[0] = fix(weights[0])
    # print("after", weights[0], weights[0].shape)

    # x = np.array(3)
    # print("before", x, x.shape)
    # x = fix(x)
    # print("after", x, x.shape)

    # def pruneSames(self):
    #     sames = [0] * len(self.data)
    #     out = []
    #     for i in range(0, len(self.data)):
    #         for j in range(i + 1, len(self.data)):
    #             sames[i] = sameList(self.data[i], self.data[j])
    #     for i in range(0, len(self.data)):
    #         if(sames[i] == 0):
    #             out.append(self.data[i])
    #     self.data = out

    # def makeRules1(self, line, lhw, rhw):
    #     print('>>>>>>>>>>>', lhw, rhw)
    #     n = len(line)
    #     if(lhw+rhw>n):
    #         return

    #     i,j,k,x = 0,0,0,0
    #     while i<n:
    #         lhs = []
    #         k = i
    #         while k<i+lhw:
    #             lhs.append(line[k%n])
    #             k = 1
    #         j = 0
    #         while j<n:
    #             rhs = []
    #             k = i+j+lhw
    #             x = 0
    #             while k<(i+j+x+lhw+rhw):
    #                 if((k%n) == i):
    #                     #print('skip',  i, j, k)
    #                     x = 1
    #                 else:
    #                     rhs.append(line[k%n])
    #                     #print(i, j, k)
    #                     #print(lhs, rhs)
    #                 k = 1
    #             if(lhw == 2 and rhw == 1):
    #                 self.addRule(lhs, rhs)
    #             j = 1
    #         i = 1
    #             # if(lhw+rhw == 2):
    #             #     self.addRule(rhs, lhs)
    #     self.makeRules1(line, lhw, rhw+1)
    #     self.makeRules1(line, lhw+1, rhw)


# ==============================================================
# Unused
# ==============================================================


class NeuralNetwork:
    def __init__(self, X, Y):
        self.w0 = np.array([[0.85857033, 0.89037339, 0.69891232, 0.83096068],
                            [0.27753788, 0.20586593, 0.30851418, 0.3868346],
                            [0.52653618, 0.5921138, 0.66015643, 0.69332094]])
        self.w1 = np.array(
            [[0.9811953], [0.25472213], [0.11068546], [0.35546831]])
        self.input = X
        # self.w0         = np.random.rand(self.input.shape[1],self.input.shape[0])

        # print("w1",self.w0)
        # print("x shape",self.input.shape)
        print("w0 shape", self.w0.shape)

        # self.w1         = np.random.rand(4, 1)
        # print("w1",self.w1)
        print("w1 shape", self.w1.shape)
        self.Y = Y
        self.out = np.zeros(self.Y.shape)
        # print("Y shape",self.Y.shape)

    def nneval(self):
        print("layeri shape", 0, self.input.shape)
        # hidden1 = W1*X
        # print("=======forward=======")

        z = np.dot(self.input, self.w0)
        # print("z shape",z.shape)

        self.layer1 = sigmoid(z)
        print("layeri shape", 1, self.layer1.shape)

        out_pre = np.dot(self.layer1, self.w1)
        self.out = sigmoid(out_pre)

    def propBack(self, i):
        # application of the chain rule to find derivative
        # of the loss function with respect to w2 and w1
        derLoss = 2 * (self.Y - self.out)
        derSig = deriv_sig(self.out)

        z2 = derSig * derLoss  # out shape
        print("z shape", 2, z2.shape)

        ders1 = np.dot(self.layer1.T, z2)
        # print("der_w2 shape",der_w2.shape)

        a1 = np.dot(z2, self.w1.T)
        print("a shape", 1, a1.shape)

        z1 = a1 * deriv_sig(self.layer1)
        print("z shape", 1, z1.shape)

        ders0 = np.dot(self.input.T, a1)
        # print("der_w1 shape",der_w1.shape)

        # print("der_w1 = ",der_w1)
        # print("der_w1 shape",der_w1.shape)
        #  update the weights with the derivative (slope) of the loss function
        self.w0 = np.add(self.w0, ders0)
        self.w1 = np.add(self.w1, ders1)
        # print("w0 shape",self.w0.shape)
        # print("w1 shape",self.w1.shape)
