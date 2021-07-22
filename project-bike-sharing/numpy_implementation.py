import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### Set self.activation_function to your implemented sigmoid function ####
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        """ Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        """
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        """ Implement forward pass here

            Arguments
            ---------
            X: features batch
        """
        ### Forward pass ###
        # Input layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        """ Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        """
        #### Implement the backward pass here ####
        ### Backward pass ###
        sigmoid_derivative = lambda x: x * (1 - x)

        # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs

        # Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error

        # Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(error, self.weights_hidden_to_output.T)
        hidden_error_term = hidden_error * sigmoid_derivative(hidden_outputs)

        # Add Weight step (input to hidden) and Weight step (hidden to output).
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]

        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """ Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records
        """

        self.weights_hidden_to_output += (
                                                 self.lr * delta_weights_h_o) / n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += (
                                                self.lr * delta_weights_i_h) / n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        """ Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        """

        #### Implement the forward pass here ####
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.5  # 0.5
hidden_nodes = 23  # 25
output_nodes = 1

# best results:
# learning rate=0.3, hidden_outputs = 23 (Training loss: 0.084 ... Validation loss: 0.185)
# learning rate=0.5, hidden_outputs = 25 (Training loss: 0.060 ... Validation loss: 0.131)
# learning rate=0.8, hidden_outputs = 20 (Training loss: 0.058 ... Validation loss: 0.142) # don't go with bigger lr!
# learning rate=0.2, hidden_outputs = 25 (Training loss: 0.156 ... Validation loss: 0.279)
# learning rate=0.5, hidden_outputs = 20 (Training loss: 0.060 ... Validation loss: 0.169)
# learning rate=0.5, hidden_outputs = 30 (Training loss: 0.059 ... Validation loss: 0.134)
# learning rate=0.3, hidden_outputs = 25 (Training loss: 0.075 ... Validation loss: 0.158)
# learning rate=0.5, hidden_outputs = 23 (Training loss: 0.054 ... Validation loss: 0.123)
