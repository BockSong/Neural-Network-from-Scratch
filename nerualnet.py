import sys
import math
import numpy as np

Debug = True
num_class = 10

class Sigmoid(object):
    def __init__(self):
        self.value = 0

    def forward(self, x):
        self.value = 1.0 / (1 + np.exp(-x))
        return self.value

    def backward(self, x, learning_rate):
        return self.value * (1 - self.value)


class linearLayer(object):
    # <input_size> feature size of the input samples.
    # <output_size> output size of each sample.
    # <init_flag> initialization mode.
    def __init__(self, input_size, output_size, init_flag):
        if (init_flag == 1): # random initialization
            self.W = np.random.uniform(size = (input_size, output_size))
            self.b = np.random.uniform(output_size)
        else: # zero initialization
            self.W = np.zeros(size = (input_size, output_size))
            self.b = np.zeros(output_size)

    # return the result of linear transformation
    # x: (input_size)
    # return: (output_size)
    def forward(self, x):
        self.x = x
        self.z = np.dot(self.x, self.W) + self.b
        return self.z

    # delta: (output_size)
    # return: (input_size)
    def backward(self, delta, learning_rate):
        # update parameters
        self.W = self.W - learning_rate * np.dot(self.x.T, delta)
        self.b = self.b - learning_rate * delta

        # TODO: return delta for last layer
        delta_last = np.dot(delta, self.W.T)
        return delta_last


class softmaxCrossEntropy(object):
    def __init__(self):
        super(softmaxCrossEntropy, self).__init__()
        self.grad = 0

    # x: (num_class)
    # y: an integer within [0, num_class - 1]
    # return: loss
    def forward(self, x, y):
        label = np.eye(x.shape)[y] # create one-hot label
        self.grad = np.zeros((x.shape))
        
        a = np.max(x) # use maximum to deal with overflow
        SumExp = np.sum(np.exp(x - a))
        self.grad = np.exp(x - a) / SumExp - label # /hat{yi} - yi
        LogSumExp = a + np.log(SumExp)
        loss = -np.sum(x * label) + np.sum(label) * LogSumExp # first * -> Hadamard product
        return loss

    # return: (num_class)
    def backward(self):
        return self.grad


class nn(object):
    def __init__(self, input_size, hidden_units, learning_rate, init_flag, metrics_out):
        self.learning_rate = learning_rate
        self.metrics_out = metrics_out
        self.layer = [
            linearLayer(input_size, hidden_units, init_flag),
            Sigmoid(),
            linearLayer(hidden_units, num_class, init_flag)
        ]
        self.criterion = softmaxCrossEntropy()

    # SGD_step: update param by taking one SGD step
    # @param
    # <feature> a 1-D numpy array
    # <label> an integer with in [0, num_class - 1]
    def SGD_step(self, feature, label):
        # pass x through layers one by one
        for layer in self.layers:
            x = layer.forward(x)
        
        # get loss by the criterion
        loss = self.criterion.forward(feature, label)

        # back propagation and update parameters
        delta = self.criterion.backward()
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)
        return loss


    def train_model(self, train_file, num_epoch):
        dataset = [] # a list of features
        # read the dataset
        with open(train_file, 'r') as f:
            for line in f:
                split_line = line.strip().split(',')
                label = int(split_line[0])
                feature = np.asarray(split_line[1:])
                #feature[len(self.dic)] = 1 # add the bias feature
                data = [label, feature]
                dataset.append(data)

        with open(metrics_out, 'w') as f_metrics:
            # perform training
            for epoch in range(num_epoch):
                train_loss = 0
                for idx in range(len(dataset)):
                    train_loss += self.SGD_step(dataset[idx][1], dataset[idx][0])
                    if Debug:
                        print("[Epoch ", epoch + 1, "] Step ", idx + 1, ", train_loss: ", train_loss)

                if Debug:
                    print("[Epoch ", epoch + 1, "] ", end='')
                    print("train_loss: ", train_loss, end=' ')
                    print("test_loss: ", self.evaluate(test_input, test_out))
                    print("train_error: ", self.evaluate(train_input, train_out), end=' ')
                    print("test_error: ", self.evaluate(test_input, test_out))

                f_metrics.write("epoch=" + epoch + " crossentryopy(train): " + train_loss + "\n")
                f_metrics.write("epoch=" + epoch + " crossentryopy(test): " + "\n")


    # predict y given an array x
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        sm = np.exp(x) / np.sum(np.exp(x), axis=0)
        return np.argmax(sm)

    # TODO: complete this func
    def evaluate(self, in_path, out_path):
        error = 0.
        total = 0.

        with open(in_path, 'r') as f_in:
            with open(out_path, 'w') as f_out:
                for line in f_in:
                    split_line = line.strip().split('\t')
                    words = dict()
                    for i in range(1, len(split_line)):
                        words[int(split_line[i].split(":")[0])] = 1
                    words[len(self.dic)] = 1 # add the bias feature

                    pred = self.predict(words)
                    if pred != int(split_line[0]):
                        error += 1
                    f_out.write(str(pred) + "\n")
                    total += 1

        return error / total # len(data)


if __name__ == '__main__':
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
    train_error = model.evaluate(train_input, train_out)
    test_error = model.evaluate(test_input, test_out)

    print("train_error: ", train_error)
    print("test_error: ", test_error)

    # Output: Metrics File
    with open(metrics_out, 'w') as f_metrics:
        f_metrics.write("error(train): " + str(train_error) + "\n")
        f_metrics.write("error(test): " + str(test_error) + "\n")
