import sys
import math
import numpy as np

Debug = True
learning_rate = 0.1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sparse_add(X, W):
    for i, v in X.items():
        W[i] += v
    return W

# not used
def sparse_sub(X, W):
    for i, v in X.items():
        W[i] -= v
    return W

def sparse_mul(X, r):
    for i in X.keys():
        X[i] = X[i] * r
    return X

def sparse_dot(X, W):
    product = 0.0
    for i, v in X.items():
        product += W[i] * v
    return product

def build_dic(dict_input):
    dic = dict()
    with open(dict_input, 'r') as f_dict:
        for line in f_dict:
            split_line = line.strip().split(' ')
            dic[split_line[0]] = split_line[1]
    return dic


class lr(object):
    def __init__(self, dic):
        self.learning_rate = learning_rate
        self.dic = dic
        self.param = dict()
        # bias is added in the end
        for i in range(len(dic) + 1):
            self.param[i] = 0 # init all params to 0

    def print_param(self):
        print("{", end = "")
        for key in self.param.keys():
            if self.param[key] != 0:
                print(key, ": [", round(self.param[key], 5), end = "] ")
        print("}")

    # SGD_step: update param by taking one SGD step
    # @param
    # <feature>: a dict
    # <label>: a binary integer
    def SGD_step(self, feature, label):
        #print("label: ", label)
        #print("feature: ", feature)
        s1 = sparse_dot(feature, self.param)
        #print("s1: ", s1)
        #mid_res = label - (1 - sigmoid(s1)) # scalar
        mid_res = label - math.exp(s1) / (1 + math.exp(s1)) # scalar
        #print("mid_res: ", mid_res)
        grad = sparse_mul(feature, mid_res * self.learning_rate) # vector
        #print("grad: ", grad)
        self.param = sparse_add(grad, self.param)
        #print("param: ")
        #self.print_param()
        # bias is updated along with W

    def train_model(self, train_file, num_epoch):
        dataset = [] # a list of features
        # read the dataset
        with open(train_file, 'r') as f:
            for line in f:
                split_line = line.strip().split('\t')
                feature = []
                feature.append(split_line[0])

                sparse_word = {}
                for i in range(1, len(split_line)):
                    sparse_word[int(split_line[i].split(":")[0])] = 1
                sparse_word[len(self.dic)] = 1 # add the bias feature
                feature.append(sparse_word)
                dataset.append(feature)

        # perform training
        for i in range(num_epoch):
            j = 0
            for feature in dataset:
                #print("Step ", j + 1, ":")
                self.SGD_step(feature[1], int(feature[0]))
                j += 1
                #print("train_error: ", self.evaluate(train_input, train_out), end=' ')
                #print("test_error: ", self.evaluate(test_input, test_out))
                #if j == 30:
                #    break

            if Debug:
                #print("param: ")
                #self.print_param()
                print("[Epoch ", i + 1, "] ", end='')
                print("train_error: ", self.evaluate(train_input, train_out), end=' ')
                print("test_error: ", self.evaluate(test_input, test_out))


    # Use lr to predict y given a list of words
    def predict(self, words):
        mid = sparse_dot(words, self.param)
        #print(mid)
        prob_posi = sigmoid(mid)
        #print(prob_posi)
        if (prob_posi > 0.5):
            #print("you predict 1")
            return 1
        else:
            #print("you predict 0")
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
    train_input = sys.argv[1]  # path to the formatted training input .tsv file
    val_input = sys.argv[2] # path to the formatted validation input .tsv file
    test_input = sys.argv[3] # path to the formatted test input .tsv file
    dict_input = sys.argv[4] # path to the dictionary input .txt file
    train_out = sys.argv[5] # path to output .labels file which predicts on trainning data
    test_out = sys.argv[6] #  path to output .labels file which predicts on test data
    metrics_out = sys.argv[7] # path of the output .txt file to write metrics
    num_epoch = int(sys.argv[8]) # an integer specifying the number of times SGD loops

    # read and build dic
    dic = build_dic(dict_input)

    model = lr(dic)

    # training: build lr model
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
