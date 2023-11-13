import numpy as np

class NN:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate, epochs):
        
        # initializing the instance variables
        self.input_neurons = input_neurons 
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.epochs = epochs
        
        # Links of weights from input layer to hidden layer
        self.wih = np.random.normal(0.0, pow(self.input_neurons, -0.5), (self.hidden_neurons, self.input_neurons))
        self.bih = 0
        
        # Links of weights from hidden layer to output layer
        self.who = np.random.normal(0.0, pow(self.hidden_neurons, -0.5), (self.output_neurons, self.hidden_neurons))
        self.bho = 0

        self.lr = learning_rate # Learning rate
        
        # Sigmoid Activation
    def activation(self, Z):
        return 1.0/(1.0 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        return self.activation(Z) * (1 - self.activation(Z))
    
    # Forward propagation
    def forward(self, input_list):
        
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs) + self.bih # (w.X) + bias Finding dot product
        hidden_outputs = self.activation(hidden_inputs) # Applying activation
        final_inputs = np.dot(self.who, hidden_outputs) + self.bho
        final_outputs = self.activation(final_inputs)
        return final_outputs
    
    # Back propagation
    def backprop(self, inputs_list, targets_list):
        
        inputs = np.array(inputs_list, ndmin=2).T
        tj = np.array(targets_list, ndmin=2).T # Targets
        # passing inputs to the hidden layer
        hidden_inputs = np.dot(self.wih, inputs) + self.bih

        # Getting outputs from the hidden layer
        hidden_outputs = self.activation(hidden_inputs)
        
        # Passing inputs from the hidden layer to the output layer
        final_inputs = np.dot(self.who, hidden_outputs) + self.bho
        
        # Getting output from the output layer
        yj = self.activation(final_inputs)
        
        # Finding the errors from the output layer
        output_errors = -(tj - yj)
        
        # Finding the error in the hidden layer
        hidden_errors = np.dot(self.who.T, output_errors)

        # Updating the weights using Update Rule
        self.who -= self.lr * np.dot((output_errors * self.sigmoid_derivative(yj)), np.transpose(hidden_outputs))
        self.wih -= self.lr * np.dot((hidden_errors * self.sigmoid_derivative(hidden_outputs)), np.transpose(inputs))


        #updating bias
        self.bho -= self.lr * (output_errors * self.sigmoid_derivative(yj))
        self.bih -= self.lr * (hidden_errors * self.sigmoid_derivative(hidden_outputs))
        pass

    # Performing Gradient Descent Optimization using Backpropagation
    def fit(self, inputs_list, targets_list):
        for epoch in range(self.epochs):         
            self.backprop(inputs_list, targets_list)
            print(f"Epoch {epoch}/{self.epochs} completed.")
            
    def predict(self, X): 
        outputs = self.forward(X).T
        return outputs

data = np.array([[0, 0, 1, 0, 1],
              [1, 1, 1, 0, 0],
              [1, 0, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [1, 0, 0, 1, 1],
              [0, 1, 0, 0, 1],
              [1, 1, 0, 1, 0],
              [0, 0, 0, 1, 0],
              [1, 0, 0, 0, 1],
              [0, 1, 1, 0, 1]])

target = np.array([[0],
              [1],
              [1],
              [0],
              [1],
              [0],
              [1],
              [0],
              [1],
              [1]])


nn = NN(data.shape[1], 10, 1, 0.1, 1000)
nn.fit(data, target)
nn.predict(data)

# output:

# array([[0.00902083],
#        [0.99684914],
#        [0.99625946],
#        [0.00701471],
#        [0.99640161],
#        [0.00922074],
#        [0.99667569],
#        [0.00483531],
#        [0.99673216],
#        [0.98974989]])