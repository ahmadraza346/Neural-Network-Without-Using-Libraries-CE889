import math
from random import random
from csv import reader
import my_constants_and_functions

# importing the variable values from my_constants_and_functions module
learning_rate = my_constants_and_functions.learning_rate
lembda_rate = my_constants_and_functions.lembda_rate
momentum_rate = my_constants_and_functions.momentum_rate
number_input = my_constants_and_functions.number_input
number_output = my_constants_and_functions.number_output
number_hidden_layer = my_constants_and_functions.number_hidden_layer
number_weights_in_input_layer = (
        number_input * number_hidden_layer)  # specifying the number of weights from input to hidden layer
number_weights_in_output_layer = (
        number_hidden_layer * number_output)  # specifying the number of weights from hidden to output layer


# this class facilitates in calculating the root mean square error for the training as well as validation data
class calculate_rmse_error:
    def __init__(self, input_layer_weights, output_layer_weights, X, Y, input_layer_bias_weights,
                 output_layer_bias_weights):
        self.input_layer_weights = input_layer_weights
        self.output_layer_weights = output_layer_weights
        self.X = X
        self.Y = Y
        self.input_layer_bias_weights = input_layer_bias_weights
        self.output_layer_bias_weights = output_layer_bias_weights
        self.values_of_hidden_layer_neurons = []
        self.values_of_output_layer_neurons = []
        # running feedforward on the data
        arb = my_constants_and_functions.feed_forward(self.input_layer_weights, self.output_layer_weights, self.X,
                                                      self.input_layer_bias_weights, self.output_layer_bias_weights)
        # computing the errors for the output layer
        sums = 0
        for i in range(number_output):
            e = float(self.Y[i]) - arb[i]
            e = e * e  # squaring errrors to calculate rmse
            sums = sums + e
            # e = round(e, 2)
        sums = sums / 2
        errorss.append(sums)  # appending the errors to a list for each iteration


class Network:  # this class facilitates in feed forward and back propagation
    def __init__(self, n_input, n_output, n_hidden_layer, size_input_layer_weights, size_output_layer_weights,
                 input_layer_weights, output_layer_weights, input_layer_delta_weights, output_layer_delta_weights,
                 X, Y, input_layer_bias_weights, output_layer_bias_weights, input_delta_bias_weights,
                 output_delta_bias_weights):

        self.n_input = n_input  # number of neurons in the input layer
        self.n_output = n_output  # number of neurons in the output layer
        self.n_hidden_layer = n_hidden_layer  # number of neurons in the hidden layer taken from user
        self.size_input_layer_weights = size_input_layer_weights  # specifying the number of weights from input to hidden layer
        self.size_output_layer_weights = size_output_layer_weights  # specifying the number of weights from hidden to output layer
        self.input_layer_weights = input_layer_weights
        self.output_layer_weights = output_layer_weights
        self.input_layer_bias_weights = input_layer_bias_weights
        self.output_layer_bias_weights = output_layer_bias_weights
        self.input_delta_bias_weights = input_delta_bias_weights
        self.output_delta_bias_weights = output_delta_bias_weights
        self.X = X
        self.Y = Y
        self.values_hidden_layer_neurons = []  # array to store the values of hidden layer neurons
        self.values_output_layer_neurons = []  # array to store the values of output layer neurons
        self.output_layer_errors = []  # array to store output layer errors
        self.output_layer_local_gradient = []
        self.hidden_layer_local_gradient = []

        self.input_layer_weights_updated = []
        self.output_layer_weights_updated = []
        self.input_bias_weights_updated = []
        self.output_bias_weights_updated = []
        self.input_layer_delta_weights = input_layer_delta_weights
        self.output_layer_delta_weights = output_layer_delta_weights

    def feed_forward(self):
        # calculating the values of hidden layer neurons
        m = 0
        i = 0
        for i in range(self.n_hidden_layer):
            n = m + self.n_hidden_layer
            z = (self.input_layer_weights[m] * float(self.X[0])) + (self.input_layer_weights[n] * float(self.X[1])) + (
                    1 * self.input_layer_bias_weights[i])
            x = self.activation_func(z)
            # x = round(x, 2)
            self.values_hidden_layer_neurons.append(x)
            m = m + 1
        # calculating the values of output layer neurons
        o = 0
        n = o
        for i in range(self.n_output):
            m = []
            m.append(n)

            for j in range(self.n_hidden_layer - 1):
                n = n + self.n_output
                m.append(n)
            r = 0
            t = 0
            b = len(m)
            for k in range(len(m)):
                r = r + (self.output_layer_weights[m[k]] * self.values_hidden_layer_neurons[t])
                t = t + 1
            r = r + (1 * self.output_layer_bias_weights[i])
            s = self.activation_func(r)
            # s = round(s, 2)
            self.values_output_layer_neurons.append(s)

            n = o + 1
        # computing the errors for the output layer
        for i in range(self.n_output):
            e = float(self.Y[i]) - self.values_output_layer_neurons[i]
            # e = round(e, 2)
            self.output_layer_errors.append(e)

    def back_propagation(self):
        # computing the local gradient for the output layer
        for i in range(self.n_output):
            lg = learning_rate * self.values_output_layer_neurons[i] * (1 - self.values_output_layer_neurons[i]) * \
                 self.output_layer_errors[i]
            # lg = round(lg, 2)
            self.output_layer_local_gradient.append(lg)
        k = 0
        # Calculating the delta weights for the output layer
        for i in range(self.n_hidden_layer):
            for j in range(self.n_output):
                a = (learning_rate * self.output_layer_local_gradient[j] * self.values_hidden_layer_neurons[i]) + (
                        momentum_rate * self.output_layer_delta_weights[k])
                # a = round(a, 2)
                self.output_layer_delta_weights[k] = a
                k = k + 1
        # calculating the delta bias weights for the output layer
        for i in range(self.n_output):
            a = (learning_rate * self.output_layer_local_gradient[i] * 1) + (
                    momentum_rate * self.output_delta_bias_weights[i])
            self.output_delta_bias_weights[i] = a
        # updating the output layer weights
        for i in range(self.size_output_layer_weights):
            a = self.output_layer_weights[i] + self.output_layer_delta_weights[i]
            # a = round(a, 2)
            self.output_layer_weights_updated.append(a)
        # updating the output layer delta weights
        for i in range(self.n_output):
            a = self.output_layer_bias_weights[i] + self.output_delta_bias_weights[i]
            # a = round(a, 2)
            self.output_bias_weights_updated.append(a)

        # computing the local gradient for the hidden layer
        sums = 0
        weight_index = 0
        for i in range(self.n_hidden_layer):
            temp = 0

            for j in range(len(self.output_layer_local_gradient)):
                temp = temp + (self.output_layer_weights[weight_index] * self.output_layer_local_gradient[j])
                weight_index = weight_index + 1

            lg = learning_rate * self.values_hidden_layer_neurons[i] * (
                    1 - self.values_hidden_layer_neurons[i]) * temp
            # lg = round(lg, 2)
            self.hidden_layer_local_gradient.append(lg)
        # calculating the delta weights for input layer
        k = 0
        for i in range(self.n_input):
            for j in range(self.n_hidden_layer):
                a = (learning_rate * self.hidden_layer_local_gradient[j] * float(self.X[i])) + (
                        momentum_rate * self.input_layer_delta_weights[k])
                # a = round(a, 2)
                self.input_layer_delta_weights[k] = a
                k = k + 1
        # calculating the bias delta weights for the input layer
        for i in range(self.n_hidden_layer):
            a = (learning_rate * self.hidden_layer_local_gradient[i] * 1) + (
                    momentum_rate * self.input_delta_bias_weights[i])
            self.input_delta_bias_weights[i] = a
        # updating the input layer weights
        for i in range(self.size_input_layer_weights):
            a = self.input_layer_weights[i] + self.input_layer_delta_weights[i]
            # a = round(a, 2)
            self.input_layer_weights_updated.append(a)
        # updating the input layer bias weights
        for i in range(self.n_hidden_layer):
            a = self.input_delta_bias_weights[i] + self.input_layer_bias_weights[i]
            self.input_bias_weights_updated.append(a)

    def activation_func(self, input_value):
        return 1 / (1 + math.exp(-lembda_rate * input_value))


local = 0
epoch_count_array = []  # keeps record of number of epochs
rmse_array = []  # saves rmse
rmse1_array = []
epoch_count = 0  # saves number of epochs
prev_root_mean_square_error_validation = 1  # pre-defined values just to run the loop
root_mean_square_error_validation = 0.9  # pre-defined values just to run the loop
while root_mean_square_error_validation < prev_root_mean_square_error_validation :  # stopping criteria
    # shuffle data after each epoch
    # ip = open('x_train.csv', 'r')
    # xd = ip.readlines()
    # shuffle(xd)
    # ip = open('y_train.csv', 'r')
    # yd = ip.readlines()
    # shuffle(yd)
    prev_root_mean_square_error_validation = root_mean_square_error_validation  # updating the prev values
    # reading the X and Y training data's first row
    with open('x_train.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        header = next(csv_reader)
    X = header
    with open('y_train.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        header = next(csv_reader)
    Y = header
    # for first epoch's first iteration assigning random values to weights
    if epoch_count == 0:
        i2h_layer_weights = []  # array to save the weights from input to the hidden layer
        for i in range(number_weights_in_input_layer):  # assigning random values to weights from input to hidden layer
            a = random()
            i2h_layer_weights.append(a)
        h2o_layer_weights = []  # array to save the weights from hidden to output layer
        for i in range(number_weights_in_output_layer):  # assigning random val to weights from hidden to output layer
            a = random()
            h2o_layer_weights.append(a)
        i2h_bias_weights = []
        for i in range(number_hidden_layer):  # assigning random values to  bias weights from input to hidden layer
            a = random()
            i2h_bias_weights.append(a)
        h2o_bias_weights = []
        for i in range(number_output):  # assigning random values to bias weights from input to hidden layer
            a = random()
            h2o_bias_weights.append(a)
    else:
        # for epochs other than the 1st epoch
        i2h_layer_weights = i2h
        h2o_layer_weights = h2o
        i2h_bias_weights = i2h_bias
        h2o_bias_weights = h2o_bias

    # delta weights
    i2h_delta_weights = []
    h2o_delta_weights = []
    i2h_bias_delta_weights = []
    h2o_bias_delta_weights = []
    # initially for an epoch, delta weights are zero in first iteration
    for i in range(number_weights_in_input_layer):
        i2h_delta_weights.append(0)
    for i in range(number_weights_in_output_layer):
        h2o_delta_weights.append(0)
    for i in range(number_hidden_layer):
        i2h_bias_delta_weights.append(0)
    for i in range(number_output):
        h2o_bias_delta_weights.append(0)
    # first iteration of an epoch
    a = Network(number_input, number_output, number_hidden_layer, number_weights_in_input_layer,
                number_weights_in_output_layer, i2h_layer_weights, h2o_layer_weights, i2h_delta_weights,
                h2o_delta_weights,
                X, Y, i2h_bias_weights, h2o_bias_weights, i2h_bias_delta_weights, h2o_bias_delta_weights)
    a.feed_forward()
    a.back_propagation()
    # reading data for rest of the iterations row by row
    with open('x_train.csv', 'r') as read_x:
        with open('y_train.csv', 'r') as read_y:
            csv_x_reader = reader(read_x)
            csv_y_reader = reader(read_y)
            x_header = next(csv_x_reader)
            x_header = next(csv_x_reader)
            y_header = next(csv_y_reader)
            y_header = next(csv_y_reader)
            sum = 1
            while sum <= 264566:  # looping on number of rows in training data
                sum = sum + 1
                X = next(csv_x_reader)
                Y = next(csv_y_reader)
                # taking weights from the previous iteration
                i2h_layer_weights = a.input_layer_weights_updated
                h2o_layer_weights = a.output_layer_weights_updated
                i2h_delta_weights = a.input_layer_delta_weights
                h2o_delta_weights = a.output_layer_delta_weights
                i2h_bias_weights = a.input_bias_weights_updated
                h2o_bias_weights = a.output_bias_weights_updated
                i2h_bias_delta_weights = a.input_delta_bias_weights
                h2o_bias_delta_weights = a.output_delta_bias_weights

                # 2nd and other iterations
                a = Network(number_input, number_output, number_hidden_layer, number_weights_in_input_layer,
                            number_weights_in_output_layer, i2h_layer_weights, h2o_layer_weights, i2h_delta_weights,
                            h2o_delta_weights,
                            X, Y, i2h_bias_weights, h2o_bias_weights, i2h_bias_delta_weights, h2o_bias_delta_weights)
                a.feed_forward()
                a.back_propagation()
    # taking final weights of an epoch for next epoch
    i2h = a.input_layer_weights_updated
    h2o = a.output_layer_weights_updated
    i2h_bias = a.input_bias_weights_updated
    h2o_bias = a.output_bias_weights_updated
    errorss = []
    # taking final weights of an epoch for rmse
    i2h_layer_weights = a.input_layer_weights_updated
    h2o_layer_weights = a.output_layer_weights_updated
    i2h_bias_weights = a.input_bias_weights_updated
    h2o_bias_weights = a.output_bias_weights_updated
    # reading the training data
    with open('x_train.csv', 'r') as read_x:
        with open('y_train.csv', 'r') as read_y:
            csv_x_reader = reader(read_x)
            csv_y_reader = reader(read_y)
            x_header = next(csv_x_reader)
            y_header = next(csv_y_reader)
            sum = 1
            while sum <= 264567:
                sum = sum + 1
                X = next(csv_x_reader)
                Y = next(csv_y_reader)
                # calculating the squared errors
                b = calculate_rmse_error(i2h_layer_weights, h2o_layer_weights,
                                         X, Y, i2h_bias_weights, h2o_bias_weights)
    summation = 0
    for i in range(len(errorss)):
        # summing errros and calculating the rmse
        summation = summation + errorss[i]
    summation = summation / len(errorss)
    root_mean_square_error_train = math.sqrt(summation)
    # keeping error to 5 decimal places for stopping criteria
    root_mean_square_error_train = round(root_mean_square_error_train, 6)
    rmse1_array.append(root_mean_square_error_train)

    errorss = []
    # calculating the rmse for validation data
    i2h_layer_weights = a.input_layer_weights_updated
    h2o_layer_weights = a.output_layer_weights_updated
    i2h_bias_weights = a.input_bias_weights_updated
    h2o_bias_weights = a.output_bias_weights_updated
    with open('x_validate.csv', 'r') as read_x:
        with open('y_validate.csv', 'r') as read_y:
            csv_x_reader = reader(read_x)
            csv_y_reader = reader(read_y)
            x_header = next(csv_x_reader)
            y_header = next(csv_y_reader)
            sum = 1
            while sum <= 56693:
                sum = sum + 1
                X = next(csv_x_reader)
                Y = next(csv_y_reader)
                c = calculate_rmse_error(i2h_layer_weights, h2o_layer_weights,
                                         X, Y, i2h_bias_weights, h2o_bias_weights)
    summation = 0
    for i in range(len(errorss)):
        summation = summation + errorss[i]
    summation = summation / len(errorss)
    root_mean_square_error_validation = math.sqrt(summation)
    root_mean_square_error_validation = round(root_mean_square_error_validation,6)

    print(root_mean_square_error_validation)
    epoch_count_array.append(epoch_count)
    rmse_array.append(root_mean_square_error_validation)
    epoch_count = epoch_count + 1
print(epoch_count)

# creating a txt file to save weights from the final epoch
with open('weights.txt', 'w') as output:
    output.write(str(c.input_layer_weights))
    output.write('\n')
    output.write(str(c.output_layer_weights))
    output.write('\n')
    output.write(str(c.input_layer_bias_weights))
    output.write('\n')
    output.write(str(c.output_layer_bias_weights))

