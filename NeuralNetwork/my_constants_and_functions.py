import math

# defining hyper-parameters
learning_rate = 0.7
lembda_rate = 0.7
momentum_rate = 0.1
number_input = 2
number_output = 2
number_hidden_layer = 8


# activation function
def activation_func(input_value):
    return 1 / (1 + math.exp(-lembda_rate * input_value))


# feedforward
def feed_forward(input_layer_weights, output_layer_weights, X, input_layer_bias_weights, output_layer_bias_weights):
    # calculating the values of hidden layer neurons
    values_of_hidden_layer_neurons = []
    values_of_output_layer_neurons = []
    m = 0
    i = 0
    for i in range(number_hidden_layer):
        n = m + number_hidden_layer
        z = (input_layer_weights[m] * float(X[0])) + (input_layer_weights[n] * float(X[1])) + (
                1 * input_layer_bias_weights[i])
        x = activation_func(z)
        values_of_hidden_layer_neurons.append(x)
        m = m + 1
    # calculating the values of output layer neurons
    o = 0
    n = o
    for i in range(number_output):
        m = []
        m.append(n)

        for j in range(number_hidden_layer - 1):
            n = n + number_output
            m.append(n)
        r = 0
        t = 0
        b = len(m)
        for k in range(len(m)):
            r = r + (output_layer_weights[m[k]] * values_of_hidden_layer_neurons[t])
            t = t + 1
        r = r + (1 * output_layer_bias_weights[i])
        s = activation_func(r)
        values_of_output_layer_neurons.append(s)
        n = o + 1
    return values_of_output_layer_neurons
