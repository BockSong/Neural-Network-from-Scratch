import sys
import math
import numpy as np

Debug = False
PrintGrad = False
num_class = 10
epsilon = 1e-5
diff_th = 1e-7

class Sigmoid(object):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.z = 1.0 / (1 + np.exp(-x))
        return self.z

    def backward(self, delta, learning_rate):
        grad = delta * (self.z * (1 - self.z))
        if PrintGrad:
            print("     Sigmoid grad: ", grad)
        return grad

class linearLayer(object):
    def __init__(self, input_size, output_size, init_flag):
        if (init_flag == 1): # random initialization
            self.W = np.random.uniform(low = -0.1, high = 0.1, size = (input_size, output_size))
            self.b = np.random.uniform(low = -0.1, high = 0.1, size = (output_size))
        else: # zero initialization
            self.W = np.zeros((input_size, output_size))
            self.b = np.zeros(output_size)

    # return the result of linear transformation
    # x: (input_size)
    # return: (output_size)
    def forward(self, x):
        self.x = x
        self.a = np.dot(self.x, self.W) + self.b
        return self.a

    # delta: (output_size)
    # return: (input_size)
    def backward(self, delta, learning_rate):
        # FIRST compute delta_last
        delta_last = np.dot(delta, self.W.T)

        # then update parameters
        dW = np.dot(self.x.reshape(self.x.shape[0], 1), delta.reshape(1, delta.shape[0]))
        if PrintGrad:
            print("     dW: ", dW)

        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * delta

        # finite difference approximation
        # TODO: buggy
        '''
        if Debug:
            grad = np.zeros(self.W.shape[0])
            for m in range(1, len(grad) + 1):
                d = np.zeros(self.W.shape[0])
                d[m] = 1
                v = np.dot(self.x, self.W + epsilon * d) + self.b
                v -= np.dot(self.x, self.W - epsilon * d) + self.b
                v /= 2 * epsilon
                grad[m] = v
            
            if np.linalg.norm(dW - grad) > diff_th:
                print("Gradient compute error!")
                print("dW: ", dW)
                print("grad: ", grad)
        '''

        return delta_last


class softmaxCrossEntropy(object):
    def __init__(self):
        super(softmaxCrossEntropy, self).__init__()

    # x: (num_class)
    # y: an integer within [0, num_class - 1]
    # return: loss
    def forward(self, x, y):
        label = np.eye(x.shape[0])[y] # create one-hot label
        self.grad = np.zeros((x.shape[0]))
        
        a = np.max(x) # use maximum to deal with overflow
        SumExp = np.sum(np.exp(x - a))
        self.grad = np.exp(x - a) / SumExp - label # /hat{yi} - yi
        LogSumExp = a + np.log(SumExp)
        loss = -np.sum(x * label) + np.sum(label) * LogSumExp # first * -> Hadamard product

        sm = np.exp(x) / np.sum(np.exp(x), axis=0)
        pred = np.argmax(sm)
        return loss, pred

    # return: (num_class)
    def backward(self):
        return self.grad


class nn(object):
    def __init__(self, input_size, hidden_units, learning_rate, init_flag, metrics_out):
        self.learning_rate = learning_rate
        self.metrics_out = metrics_out
        self.layers = [
            linearLayer(input_size, hidden_units, init_flag),
            Sigmoid(),
            linearLayer(hidden_units, num_class, init_flag)
        ]
        self.criterion = softmaxCrossEntropy()

    # SGD_step: update params by taking one SGD step
    # <x> a 1-D numpy array
    # <y> an integer within [0, num_class - 1]
    def SGD_step(self, x, y):
        # perform forward propogation and compute intermediate results
        if PrintGrad:
            print("		Begin forward pass")
        for layer in self.layers:
            x = layer.forward(x)
            if PrintGrad:
                print("     output: ", x)
        loss, _ = self.criterion.forward(x, y)
        if PrintGrad:
            print("			Cross entropy: ", loss)
            print("		Begin backward pass")

        # perform back propagation and update parameters
        delta = self.criterion.backward()
        if PrintGrad:
            print("			d(loss)/d(softmax inputs): ", delta)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)
            if PrintGrad:
                print("     delta: ", delta)

        if PrintGrad:
            print("			New first layer weights: ", self.layers[0].W)
            print("			New second layer weights: ", self.layers[2].W)
        return loss


    def train_model(self, train_file, num_epoch):
        dataset = [] # a list of features
        # read the dataset
        with open(train_file, 'r') as f:
            for line in f:
                split_line = line.strip().split(',')
                y = int(split_line[0])
                x = np.asarray(split_line[1:], dtype=int)
                #feature[len(self.dic)] = 1 # add the bias feature
                dataset.append([y, x])

        with open(metrics_out, 'w') as f_metrics:
            # perform training
            for epoch in range(num_epoch):
                loss = 0
                for idx in range(len(dataset)):
                    loss = self.SGD_step(dataset[idx][1], dataset[idx][0])
                    if Debug and (idx % 1000 == 0):
                        print("[Epoch ", epoch + 1, "] Step ", idx + 1, ", current_loss: ", loss)

                train_loss, train_error = self.evaluate(train_input, train_out)
                test_loss, test_error = self.evaluate(test_input, test_out)

                if Debug:
                    print("[Epoch ", epoch + 1, "] ", end='')
                    print("train_loss: ", train_loss, end=' ')
                    print("train_error: ", train_error)
                    print("test_loss: ", test_loss, end=' ')
                    print("test_error: ", test_error)

                f_metrics.write("epoch=" + str(epoch) + " crossentryopy(train): " + str(train_loss) + "\n")
                f_metrics.write("epoch=" + str(epoch) + " crossentryopy(test): " + str(test_loss) + "\n")

    # predict y given an array x
    # not used
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        sm = np.exp(x) / np.sum(np.exp(x), axis=0)
        return np.argmax(sm)

    def evaluate(self, in_path, out_path, write = False):
        total_loss, error, total = 0., 0., 0.

        with open(in_path, 'r') as f_in:
            with open(out_path, 'a') as f_out:
                for line in f_in:
                    split_line = line.strip().split(',')
                    y = int(split_line[0])
                    x = np.asarray(split_line[1:], dtype=int)

                    for layer in self.layers:
                        x = layer.forward(x)
                    loss, pred = self.criterion.forward(x, y)

                    total_loss += loss
                    if pred != y:
                        error += 1
                    if write:
                        f_out.write(str(pred) + "\n")
                    total += 1

        return total_loss / total, error / total


if __name__ == '__main__':
    if len(sys.argv) != 10:
        print("The number of command parameters is incorrect.")
        exit(-1)

    train_input = sys.argv[1]  # path to the training input .csv file
    test_input = sys.argv[2] # path to the test input .csv file
    train_out = sys.argv[3] # path to output .labels file which predicts on trainning data
    test_out = sys.argv[4] #  path to output .labels file which predicts on test data
    metrics_out = sys.argv[5] # path of the output .txt file to write metrics
    num_epoch = int(sys.argv[6]) # an integer specifying the number of times BP loops
    hidden_units = int(sys.argv[7]) # positive integer specifying the number of hidden units
    init_flag = int(sys.argv[8]) # an integer specifying whether to use RANDOM or ZERO initialization
    learning_rate = float(sys.argv[9]) # float value specifying the learning rate for SGD

    # get input_size
    with open(train_input, 'r') as f_in:
        line = f_in.readline()
        split_line = line.strip().split(',')
        input_size = len(split_line) - 1

    # build and init 
    model = nn(input_size, hidden_units, learning_rate, init_flag, metrics_out)

    # training
    model.train_model(train_input, num_epoch)

    # testing: evaluate and write labels to output files
    train_loss, train_error = model.evaluate(train_input, train_out, True)
    test_loss, test_error = model.evaluate(test_input, test_out, True)

    print("train_loss: ", train_loss, end=' ')
    print("train_error: ", train_error)
    print("test_loss: ", test_loss, end=' ')
    print("test_error: ", test_error)
    
    # Output: Metrics File
    with open(metrics_out, 'a') as f_metrics:
        f_metrics.write("error(train): " + str(train_error) + "\n")
        f_metrics.write("error(test): " + str(test_error) + "\n")
