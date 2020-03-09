import sys
import math
import numpy as np

Debug = True
num_class = 10

class Sigmoid(object):
    def __init__(self):
        self.value = 0

    def forward(self, x):
        self.value = 1.0 / (1 + math.exp(-x))
        return self.value

    def backward(self, x):
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
        self.activations = Sigmoid()

    # TODO: change to SGD rather than mini-batch
    # return the result of linear transformation
    # x: (batch_size, input_size)
    # return: (batch_size, output_size)
    def forward(self, x):
        self.batch_size = x.shape[0]
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.A = self.activations.forward(self.z)
        return self.A

    # update internal parameters and return delta for last layer
    # delta: (batch_size, output_size)
    # return: (batch_size, input_size)
    def backward(self, delta, learning_rate):
        self.dz = self.activations.backward(delta, learning_rate)
        self.dA = np.dot(self.dz, self.W.T)
        self.W = self.W - learning_rate * np.dot(self.x.T, self.dz)
        self.b = self.b - learning_rate * np.sum(self.dz, axis=0)
        return self.dA


class softmaxCrossEntropy(object):
    def __init__(self):
        super(softmaxCrossEntropy, self).__init__()
        self.sm = 0

    # x: (batch_size, n_classes)
    # y: (batch_size,), each item within [0, n_classes-1]
    # return: (2,), sum of loss and sum of accuracy of all samples
    def forward(self, x, y):
        self.logits = x
        self.labels = np.eye(x.shape[1])[y]
        self.batch_size = self.logits.shape[0]
        loss = np.zeros((self.logits.shape[0]))
        self.sm = np.zeros((self.logits.shape))
        
        for i in range(x.shape[0]):
            a = np.max(self.logits[i]) # use maximum to deal with overflow
            SumExp = np.sum(np.exp(self.logits[i] - a))
            self.sm[i] = np.exp(self.logits[i] - a) / SumExp - self.labels[i] # /hat{yi} - yi
            LogSumExp = a + np.log(SumExp)
            loss[i] = -np.sum(self.logits[i] * self.labels[i]) + np.sum(self.labels[i]) * LogSumExp
            if (np.argmax(self.logits[i]) == y[i]):
                print("correct")
        return np.sum(loss)

    # return delta for last layer
    # return: (batch_size, n_classes)
    def backward(self):
        return self.sm


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
    # <label> an integer with in [0, 9]
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
        mid = sparse_dot(words, self.param)
        prob_posi = sigmoid(mid)
        if (prob_posi > 0.5):
            return 1
        else:
            return 0

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
