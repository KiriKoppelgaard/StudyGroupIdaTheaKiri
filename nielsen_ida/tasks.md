# Data Science - Week 6

### Goals

This week's assignments are about understanding the intricacies of convolutional neural networks and ways to train them.

### Assignments

Have a stab at some out of the final set of problems in [chapter 6 of Nielsen's book](http://neuralnetworksanddeeplearning.com/chap6.html). I've copied them in here below. Don't feel obliged to do all of them - use this as an opportunity to play around with the code and deepen your understanding.

- At present, the `SGD` method requires the user to manually choose the number of epochs to train for. Earlier in the book we discussed an automated way of selecting the number of epochs to train for, known as early stopping. Modify `network3.py` to implement [early stopping](http://neuralnetworksanddeeplearning.com/chap3.html#early_stopping).
- Add a `Network` method to return the accuracy on an arbitrary data set.
- Modify the `SGD` method to allow the learning rate η to be a function of the epoch number. *Hint: After working on this problem for a while, you may find it useful to see the discussion at [this link](https://groups.google.com/forum/#!topic/theano-users/NQ9NYLvleGc).*
- Earlier in the chapter I described a technique for expanding the training data by applying (small) rotations, skewing, and translation. Modify `network3.py` to incorporate all these techniques. *Note: Unless you have a tremendous amount of memory, it is not practical to explicitly generate the entire expanded data set. So you should consider alternate approaches.*
- Add the ability to load and save networks to `network3.py`.
- A shortcoming of the current code is that it provides few diagnostic tools. Can you think of any diagnostics to add that would make it easier to understand to what extent a network is overfitting? Add them.
- We've used the same initialization procedure for rectified linear units as for sigmoid (and tanh) neurons. Our [argument for that initialization](http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization) was specific to the sigmoid function. Consider a network made entirely of rectified linear units (including outputs). Show that rescaling all the weights in the network by a constant factor c>0 simply rescales the outputs by a factor c^(L−1), where L is the number of layers. How does this change if the final layer is a softmax? What do you think of using the sigmoid initialization procedure for the rectified linear units? Can you think of a better initialization procedure? *Note: This is a very open-ended problem, not something with a simple self-contained answer. Still, considering the problem will help you better understand networks containing rectified linear units.*
- Our [analysis](http://neuralnetworksanddeeplearning.com/chap5.html#what's_causing_the_vanishing_gradient_problem_unstable_gradients_in_deep_neural_nets) of the unstable gradient problem was for sigmoid neurons. How does the analysis change for networks made up of rectified linear units? Can you think of a good way of modifying such a network so it doesn't suffer from the unstable gradient problem? *Note: The word good in the second part of this makes the problem a research problem. It's actually easy to think of ways of making such modifications. But I \[Nielsen\] haven't investigated in enough depth to know of a really good technique.*
