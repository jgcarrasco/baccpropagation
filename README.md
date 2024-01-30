# baccpropagation
Implementing neural networks in C. The idea of this project is to (i) learn C and (ii) get a solid understanding of the
fundamental concepts of backpropagation. I expect the code to be extremely cursed, but I think that it is a fun idea!
The codebase is strongly inspired on [Andrej Karpathy's Micrograd](https://github.com/karpathy/micrograd) (thanks
Andrej!)

- `engine.c`: Ultra-basic setup of a backprop engine, where each value is represented by a node and the model is
  represented by a computational graph (such as in micrograd).

# TO-DO

- [X] Training and implementing a single neuron
- [ ] Training a 1-Layer MLP
- [ ] Training a 2-Layer MLP

## Ideas

- Build a basic 1-layer NN to classify digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). I should
  achieve around $88%$ accuracy. 



