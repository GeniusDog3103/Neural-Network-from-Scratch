# %% [markdown]
# NEURAL NET FROM SCRATCH:
# -GeniusDog3103
# 
# Made in python, with numpy for arrays, torchvision for importing datasets and splitting datasets,

# %% [markdown]
# Import libraries

# %%

%pip install torchvision --index-url https://download.pytorch.org/whl/cu132
%pip install tqdm
%pip install Pillow

from torchvision import datasets
import numpy as np

# %% [markdown]
# Import MNIST dataset (hand-drawn numbers) and split it for training and testing

# %%
train_mnist_db = datasets.MNIST(root='mnist_train',train=True, download=True)#images are of matrices: (28,28)
test_mnist_db = datasets.MNIST(root='mnist_test',train=False, download=True)#images are of matrices: (28,28)

# %%
#visualize first sample's size, type
print(type(train_mnist_db[0][0]))
print(type(train_mnist_db[0][1]))
print(train_mnist_db[0][0].size)

# %% [markdown]
# Define the Entire NeuralNet class

# %%
class NeuralNet:
    def __init__(self,bias: bool, hidden_layers: int, layer_size: int, input_neurons: int, output_neurons: int, learning_rate: float, l2_regularization: float):
        self.hidden_layers = hidden_layers
        self.bias=bias #does nothing, we use biases no matter what
        self.learning_rate = learning_rate
        self.l2_regularization_rate = l2_regularization
        self.layer_size = layer_size

        self.weight_layers=[] #a list of numpy arrays(neurons in layer, inputs per neuron), which hold weights

        self.momentum_ma_optimizer_weights = [] #the momentum tracker part of AdamW, initalized as zeros
        self.momentum_ma_optimizer_biases = [] #the momentum tracker part of AdamW, initalized as zeros
        self.rmsprop_ma_optimizer_weights = [] #the rmsprop tracker part of AdamW, initalized as zeros
        self.rmsprop_ma_optimizer_biases = [] #the rmsprop tracker part of AdamW, initalized as zeros

        self.optimizer_idx = 1 #idx for counting how many times we've already optimized, which we use for bias correction code/formula

        rng = np.random.default_rng()
        for layer in range(hidden_layers):
            if len(self.weight_layers) == 0:
                self.weight_layers.append(rng.uniform(-(np.sqrt((6/input_neurons))),(np.sqrt((6/input_neurons))),(layer_size, input_neurons)))

                self.momentum_ma_optimizer_weights.append(np.zeros((layer_size,input_neurons)))
                self.rmsprop_ma_optimizer_weights.append(np.zeros((layer_size,input_neurons)))
            else:
                self.weight_layers.append(rng.uniform(-(np.sqrt((6/self.weight_layers[-1].shape[0]))),(np.sqrt((6/self.weight_layers[-1].shape[0]))), (layer_size,self.weight_layers[-1].shape[0])))
                self.momentum_ma_optimizer_weights.append(np.zeros((layer_size,self.weight_layers[-1].shape[0])))
                self.rmsprop_ma_optimizer_weights.append(np.zeros((layer_size,self.weight_layers[-1].shape[0])))

                
        self.weight_layers.append(rng.uniform(-(np.sqrt((6/self.weight_layers[-1].shape[0]))),(np.sqrt((6/self.weight_layers[-1].shape[0]))),(output_neurons,self.weight_layers[-1].shape[0]))) #append the output layer for neuron weights
        self.momentum_ma_optimizer_weights.append(np.zeros((self.weight_layers[-1].shape[0], layer_size)))
        self.rmsprop_ma_optimizer_weights.append(np.zeros((self.weight_layers[-1].shape[0], layer_size)))
                
        self.bias_layers = [] #list holding 1D numpy arrays, which hold the biases in each layer. the 
        for layer in range(hidden_layers):
            self.bias_layers.append(np.zeros(layer_size,))
            self.momentum_ma_optimizer_biases.append(np.zeros(layer_size))
            self.rmsprop_ma_optimizer_biases.append(np.zeros(layer_size))

        #print(f'shape of each bias layer in hidden layers: {rng.uniform(-1,1,(layer_size,)).shape}')

        self.bias_layers.append(np.zeros(output_neurons,))
        self.momentum_ma_optimizer_biases.append(np.zeros(output_neurons))
        self.rmsprop_ma_optimizer_biases.append(np.zeros(output_neurons))
            
        self.running_cost = []
        self.running_output_grads = []
        self.activations = []
        self.last_layer_activations = [] #need to keep seperate cuz it causes errors when doing multiplication, etc.
        self.inputs = [] #needed for finding the gradients of the weights of the first hidden layer

        self.weight_grads = [] #finished grads for weights in specific layers
        self.bias_grads = [] #finished grads for biases in each layer
        
    def batch_reset(self)->None:
        """Reset the self parameters for the next training batch"""
        self.activations = []
        self.last_layer_activations = [] #need to keep seperate cuz it causes errors when doing multiplication, etc.
        self.inputs = [] #needed for finding the gradients of the weights of the first hidden layer
        self.running_cost = []
        self.running_output_grads = []

    def forprop(self,image_target:tuple)->int | None:
        """
        Forward propogation (running the model on an image)	
        Input:
                tuple: image pixels (in PIL.Image format), target
        """	
        activations_sample= []
        image = np.divide(np.asarray(image_target[0], dtype=np.float64).flatten(), 255)
        self.inputs.append(image) #divide by 255 to shrink to a range of 0 to 1.0


        previous_layer_output = np.fmax(np.zeros(self.layer_size),((self.weight_layers[0]@ image.T)+self.bias_layers[0])) #output is (num_neurons, 1)#@ is for dot-product multiplication
        copy = previous_layer_output.copy()
        activations_sample.append(copy)
        
        for num in range(1,len(self.weight_layers)-1): #we go from the first hidden layer to the last layer before output layer
            previous_layer_output = (self.weight_layers[num]@previous_layer_output)+self.bias_layers[num] #previous_layer_output = (num_neurons, 1) #@ is for dot-product multiplication
            previous_layer_output = np.fmax(np.zeros_like(previous_layer_output),previous_layer_output) #relu
            copy = previous_layer_output.copy()
            activations_sample.append(copy)
        
        previous_layer_output = (self.weight_layers[-1]@previous_layer_output)+self.bias_layers[-1]
        
        assert isinstance(self.last_layer_activations, list)
        self.last_layer_activations.append(previous_layer_output)

        if len(previous_layer_output) != 10:
            print(f'big error, cost cannot be found cuz output neurons is not 10: {len(previous_layer_output)}')
        #remember target is of shape (10,1), while previous_layer_output = (num_neurons, 1)	
        else:
            target = np.zeros(10)
            target[image_target[1]]=1
            sample_cost = np.mean(np.square(previous_layer_output-target)) 
            running_output_grad = 2*(previous_layer_output-target)
            self.running_cost.append(sample_cost) #add to running cost, we will find the average in calculate_grad()
            self.running_output_grads.append(running_output_grad)

            self.activations.append(np.array(activations_sample))

            return int(np.argmax(previous_layer_output))

    def calculate_grad(self, BATCH_SIZE): 
        batch_cost = np.sum(self.running_cost)/(len(self.running_cost))
        print(f'batch cost scalar: {batch_cost}')
        self.running_cost=[]

        self.activations = np.array(self.activations)
        self.last_layer_activations = np.array(self.last_layer_activations) #of shape=(samples,10)
        output_grads = np.array(self.running_output_grads) #.shape=(samples, output_neurons)  
        self.running_output_grads = []
        a_z_grad_func = lambda activations: np.where(activations>0,1,0)
        a_z_grads = a_z_grad_func(self.activations) #a_z_grads.shape = (samples, layers, neurons_in_layer), self.activations.shape = (samples, layers, neurons_in_layer)
        
        layer_L_grads = output_grads.T #* is for element-wise multiplication; #these grads are the gradient of the cost function in relation to z of the last layer. dot-product multiply these by the gradient of z_j in relation to a specific weight, activation, or bias to get actual gradients; #their shape is (10 output neurons, samples) 


        past_grads = [] #grads used for calculating next layer's grads
        self.weight_grads = [] #finished grads for weights in specific layers
        self.bias_grads = [] #finished grads for biases in each layer
        for l in reversed(range(self.activations.shape[1]+1)): #shape[1]=number of hidden layers, we add 1 to account for the output_layer
            #l is a layer of shape (num of neurons in layer, previous layer size)
            if isinstance(past_grads,list):
                layer_grads = (layer_L_grads @ self.activations[:,l-1,:])/len(output_grads) #see why on the eink tablet; hopefully layer_grads is of shape (10,16)

                past_grads = layer_L_grads.T #needed because we cannot use layer_grads for calculating next layer's grads, because those are the gradients of C to w_jk, NOT C to a_l-1; #we apply .T because we need (samples, neurons) for calculations for next layers
                self.weight_grads.append(layer_grads) #finished grads for output layer weight
                self.bias_grads.append((layer_L_grads @ np.ones(BATCH_SIZE))/len(output_grads)) #we multiply these to get a vector of shape (output_neurons,1)#remember we are multiplying by 1, because the gradients of z to the bias is 1 #10 because 10 neurons in output layer
                #could optimize not too include if else by adding this logic before for loop
            elif l==0: #separated because for the first hidden layer of the neural net we will have to use the input dimensions as the partial derivatives of z to w_jk as z=w_jk*x_1+w_jk*x_2+...+b; this is different from other layers which just use self.activations for the L-1 activations
            #hopefully past_grads is of shape (samples, neurons); self.weight_layers[l+1] should be of shape (neurons in last layer, neurons in this layer); a_z_grads should be of shape (samples, neurons in this layer(16))
                past_grads = (past_grads @ self.weight_layers[l+1] * a_z_grads[:,l,:]) 

                layer_grads = (past_grads.T @ np.array(self.inputs))/len(output_grads)

                self.weight_grads.append(layer_grads)
                self.bias_grads.append(past_grads.T @ np.ones(BATCH_SIZE)/len(output_grads))
            else:
                #hopefully past_grads is of shape (samples, neurons); self.weight_layers[l+1] should be of shape (neurons in last layer, neurons in this layer); a_z_grads should be of shape (samples, neurons in this layer(16))
                past_grads = (past_grads @ self.weight_layers[l+1] * a_z_grads[:,l,:])

                #we do past_grads.T to hopefully get (neurons in last layer, samples), which we will multiply by self.activations[:,l-1,:] which should be (samples, neurons in this layer)
                #print(f'shape of  self.activations[:,{l}-1,:]: {self.activations[:,l-1,:].shape}')
                layer_grads = (past_grads.T @ self.activations[:,l-1,:])/len(output_grads) #layer_grads is hopefully of shape (neurons in last layer, neurons in this layer)

                self.weight_grads.append(layer_grads)
                self.bias_grads.append(past_grads.T @ np.ones(BATCH_SIZE)/len(output_grads))
        
        self.batch_reset()

    def optimize(self):
        """
        Takes and optimizes weights and biases according to AdamW algorithm using:
        self.weight_grads = [] #finished grads for weights in specific layers; of shape (4 layers (last->first), neurons in this layer, neurons in last layer)
        self.bias_grads = [] #finished grads for biases in each layer; of shape(4 layers (last->first), biases in this layer)
        self.weight_layers #actual weights; of shape (4 layers, neurons in this layer, neurons in last layer)
        self.bias_layers #actual biases; of shape (4 layers , biases in this layer)
        self.momentum_ma_optimizer_weights #the momentum tracker part of AdamW, initalized at 0
        self.momentum_ma_optimizer_biases #the momentum tracker part of AdamW, initalized at 0
        self.rmsprop_ma_optimizer_weights #the rmsprop tracker part of AdamW, initalized at 0
        self.rmsprop_ma_optimizer_biases #the rmsprop tracker part of AdamW, initalized at 0
        """
        #hyperparemeters:
        impact_of_past_momentum_grads = 0.9 #how much we give weight/impact to past gradients for momentum ma compared to new grads; β_1
        impact_of_past_squared_grads = 0.999 #how much we give weight/impact to past gradients for RMSProp ma compared to new grads; β_2

        #reverse order of grads for the calculations as they are structured from backpropogation order by default
        self.weight_grads.reverse()
        self.bias_grads.reverse()

        #need to keep track of these 4 moving averages matrices
        element_wise_addition = lambda x,y: [np.add(x[layer_idx],y[layer_idx]) for layer_idx in range(len(x))]

        #just some error checks to ensure that we can use a singular index for layer indexation in next block of code
        assert len(self.momentum_ma_optimizer_weights)==len(self.weight_grads), f"self.momentum_ma_optimizer_weights: {len(self.momentum_ma_optimizer_weights)};\tself.weight_grads: {len(self.weight_grads)}"
        assert len(self.momentum_ma_optimizer_biases)==len(self.bias_grads), f"self.momentum_ma_optimizer_biases: {len(self.momentum_ma_optimizer_biases)};\tself.weight_grads: {len(self.bias_grads)}"
        assert len(self.rmsprop_ma_optimizer_weights)==len(self.weight_grads), f"self.rmsprop_ma_optimizer_weights: {len(self.rmsprop_ma_optimizer_weights)};\tself.weight_grads: {len(self.weight_grads)}"
        assert len(self.rmsprop_ma_optimizer_biases)==len(self.bias_grads), f"self.rmsprop_ma_optimizer_biases: {len(self.rmsprop_ma_optimizer_biases)};\tself.weight_grads: {len(self.bias_grads)}"
        
        self.momentum_ma_optimizer_weights = element_wise_addition([layer*impact_of_past_momentum_grads for layer in self.momentum_ma_optimizer_weights],[layer*(1-impact_of_past_momentum_grads) for layer in self.weight_grads]) #.reverse() to reverse the layer arrangement so that the first layer is in the front.
        self.momentum_ma_optimizer_biases = element_wise_addition([layer*impact_of_past_momentum_grads for layer in self.momentum_ma_optimizer_biases],[layer*(1-impact_of_past_momentum_grads) for layer in self.bias_grads]) #output is of shape (4 layers (first->last),biases)
        self.rmsprop_ma_optimizer_weights = element_wise_addition([layer*impact_of_past_squared_grads for layer in self.rmsprop_ma_optimizer_weights],[np.square(layer)*(1-impact_of_past_squared_grads) for layer in self.weight_grads]) #output is of shape (4 layers (first->last), neurons in this layer, neurons in last layer)
        self.rmsprop_ma_optimizer_biases = element_wise_addition([layer*impact_of_past_squared_grads for layer in self.rmsprop_ma_optimizer_biases],[np.square(layer)*(1-impact_of_past_squared_grads) for layer in self.bias_grads]) #output is of shape (4 layers (first->last),biases)


        #bias correction for both all the averages:
        unbiased_momentum_weights_grads = [(layer/(1-(impact_of_past_momentum_grads**self.optimizer_idx))) for layer in self.momentum_ma_optimizer_weights] #confused? check bias correction on paper plus the above code and the shapes; math for why this solves bias on tablet
        unbiased_momentum_bias_grads = [(layer/(1-(impact_of_past_momentum_grads**self.optimizer_idx))) for layer in self.momentum_ma_optimizer_biases] #confused? check bias correction on paper plus the above code and the shapes
        unbiased_rmsprop_weights_grads = [(layer/(1-(impact_of_past_squared_grads**self.optimizer_idx))) for layer in self.rmsprop_ma_optimizer_weights] #confused? check bias correction on paper plus the above code and the shapes
        unbiased_rmsprop_bias_grads = [(layer/(1-(impact_of_past_squared_grads**self.optimizer_idx))) for layer in self.rmsprop_ma_optimizer_biases] #confused? check bias correction on paper plus the above code and the shapes

        #just some error checks to ensure that we can use a singular index for layer indexation in next block of code
        assert len(unbiased_momentum_weights_grads)==len(self.weight_layers)
        assert len(unbiased_momentum_bias_grads)==len(self.bias_layers)
        assert len(unbiased_rmsprop_weights_grads)==len(self.weight_layers)
        assert len(unbiased_rmsprop_bias_grads)==len(self.bias_layers)
        
        
        #actual updates to the weights: #the 0.0001 value is just the epsilon to prevent division by 0
        #self.weight_layers = [self.weight_layers[layer]-(self.learning_rate*(unbiased_momentum_weights_grads[layer]/((np.sqrt(unbiased_rmsprop_weights_grads[layer])+0.0000001)+(self.l2_regularization_rate*self.weight_layers[layer])))) for layer in range(len(self.weight_layers))] #explained in paper
        #self.bias_layers = [self.bias_layers[layer]-(self.learning_rate*(unbiased_momentum_bias_grads[layer]/((np.sqrt(unbiased_rmsprop_bias_grads[layer])+0.0000001)+(self.l2_regularization_rate*self.bias_layers[layer])))) for layer in range(len(self.bias_layers))] #explained in paper
        #2nd approach i saw at cornell for the above part
        self.weight_layers = [(self.weight_layers[layer]-(self.learning_rate*(unbiased_momentum_weights_grads[layer]/(np.sqrt(unbiased_rmsprop_weights_grads[layer])+0.0000001))))-(self.learning_rate*self.l2_regularization_rate*self.weight_layers[layer]) for layer in range(len(self.weight_layers))] #explained in paper
        self.bias_layers = [(self.bias_layers[layer]-(self.learning_rate*(unbiased_momentum_bias_grads[layer]/(np.sqrt(unbiased_rmsprop_bias_grads[layer])+0.0000001))))-(self.learning_rate*self.l2_regularization_rate*self.bias_layers[layer]) for layer in range(len(self.bias_layers))] #explained in paper


        #reset parameters
        self.weight_grads = []
        self.bias_grads = []

        #update idx for counting how many times we've already optimized, which we use for bias correction code/formula
        self.optimizer_idx+=1

    def dev_debug_grads(self):
        #wanna make sure that the gradinets aren't exploding or vanishing
        #i read somewhere that you should just see the mean and std of the weights in each layer, and it should be 1 for std

        mean_weight_grad = [np.mean(layer.flatten()) for layer in self.weight_grads]
        for i in range(len(mean_weight_grad)):
            print(f"Avg grad for layer{i} (might be reversed order): {mean_weight_grad[i]}")
	


# %% [markdown]
# Lets first test forward-propagation (just running the model) with randomly initialized values. We will also show what we expected vs actual prediction.

# %%
BIAS=True #we don't have any code for if it is false
HIDDEN_LAYERS = 5
LAYER_SIZE = 16
INPUT_NEURONS=784 #cuz mnist is 28*28
OUTPUT_NEURONS=10
LEARNING_RATE = 0.0005
L2_REGULARIZATION = 0.01
BATCH_SIZE=64
EPOCHS=5

nn = NeuralNet(BIAS,HIDDEN_LAYERS,LAYER_SIZE,INPUT_NEURONS,OUTPUT_NEURONS,LEARNING_RATE,L2_REGULARIZATION)

first_image = train_mnist_db[0] #get first image (image, target)

print(f"dimensions of flattened first sampel image: {np.asarray(first_image[0]).flatten().shape}")
print(f"target for first sample: {first_image[1]}")

nn.forprop(first_image)
nn.batch_reset()

# %% [markdown]
# Loop for full train dataset forward propogation

# %%
counter =0
for image in train_mnist_db:
    print(f"target for {counter} sample: {image[1]}")
    nn.forprop(image)
    counter+=1

nn.batch_reset()

# %% [markdown]
# Loop for batch training:

# %%

for epoch in range(EPOCHS):
    samples = 0
    for image, target in train_mnist_db:
        nn.forprop((image,target))
        samples+=1
        if samples == BATCH_SIZE:
        #    nn.calculate_grad(BATCH_SIZE)
        #    nn.optimize()
            nn.batch_reset()
            break
    break


# %% [markdown]
# Code for calculate_grad() after one batch

# %%
nn.batch_reset()

for epoch in range(EPOCHS):
    samples = 0
    for image, target in train_mnist_db:
        nn.forprop((image,target))
        samples+=1
        if samples == BATCH_SIZE:
            print(len(nn.last_layer_activations))
            print(np.array(nn.last_layer_activations).shape)
            nn.calculate_grad(BATCH_SIZE)
            print(f'self.weight_grads length: {len(nn.weight_grads)}')
            print(f'self.bias_grads length: {len(nn.bias_grads)}')
        #   nn.optimize()
            #nn.batch_reset()
            break
    break


# %% [markdown]
# Code for optimize() after one batch

# %%
nn.batch_reset()

for epoch in range(EPOCHS):
    samples = 0
    for image, target in train_mnist_db:
        nn.forprop((image,target))
        samples+=1
        if samples == BATCH_SIZE:
            print(len(nn.last_layer_activations))
            print(np.array(nn.last_layer_activations).shape)
            nn.calculate_grad(BATCH_SIZE)
            print(f'self.weight_grads length: {len(nn.weight_grads)}')
            print(f'self.bias_grads length: {len(nn.bias_grads)}')
            print("*"*5)
            for layer in nn.bias_grads:
                print(layer.shape)
            print("-"*5)
            for layer in nn.momentum_ma_optimizer_biases:
                print(layer.shape)
            print("-"*5)
            for layer in nn.weight_grads:
                print(layer.shape)
            print("-"*5)
            for layer in nn.momentum_ma_optimizer_weights:
                print(layer.shape)
            print("-"*5)
            
            nn.optimize()
            #nn.batch_reset()
            break
    break

#problem is with the unbiased_... variables and the self.weight_grads in self.optimize(). make print statements for the shapes to find what and where the problem is


# %% [markdown]
# Code for full training loop with batch training

# %%
total_samples=0
num_correct = 0
for epoch in range(EPOCHS):
    samples = 0
    nn.batch_reset()
    for image, target in train_mnist_db:
        number = nn.forprop((image,target))
        samples+=1
        total_samples+=1
        if number == int(target):
            num_correct+=1
        if samples == BATCH_SIZE:
            nn.calculate_grad(BATCH_SIZE)
            #nn.dev_debug_grads()
            
            nn.optimize() #investigate_ts
            nn.batch_reset()
            samples=0

print(f'total samples: {total_samples}')
print(f'total training accuracy: {num_correct/total_samples}')

# %% [markdown]
# Performance on test database

# %%
samples = 0
num_correct = 0
for image,target in train_mnist_db:
    #print(np.array(image))
    #break
    number = nn.forprop((image,target))
    samples+=1
    if int(target)==number:
        num_correct+=1

print(f"percentage correct: {num_correct/samples}")
nn.batch_reset()


# %% [markdown]
# Custom number drawing. Draw it in paint and then download and input the url to the image. Make sure the image is in the same folder as this file.

# %%

from PIL import Image, ImageOps

def draw_custom_number():
    print("have you saved a 28x28 pixel image in Paint?")
    url = input("Input the url to the image you saved: ")
    with Image.open(fp=url, formats=['PNG']) as image:
        image = image.convert(mode='L')#try 'L' if it doesn't work
        image =  ImageOps.invert(image)
        #print(f"image.size: {image.size}")
        #print(len(np.asarray(image, dtype=np.float64).flatten()))
        #print(np.asarray(image))
        prediction = nn.forprop((image, 0)) #put 0 for the target cuz we don't need it, we wil batch_reset to not affect anything
        nn.batch_reset()
        
        print(f"Model's predicted number: {prediction}")
draw_custom_number()
nn.batch_reset()


