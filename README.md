# Machine-Learning

## SGD w/ no optimizations
```tex
Iteration [0]
Accuracy: 87.7%
Iteration [10]
Accuracy: 92.7%
...
Iteration [490]
Accuracy: 91.3%
Iteration [500]
Accuracy: 91.6%
```
As you can see nothing really changed past the first few iterations, our final result was even worse than what we started on, this is most likely due to a too high learning rate of $1.0$. Our next step will be to have the learning rate be a function that changes as we train our neural net.

## SGD w/ Changing learning rate
```tex
Iteration [0]
Accuracy: 79.3%
Iteration [10]
Accuracy: 81.0%
...
Iteration [490]
Accuracy: 93.8%
Iteration [500]
Accuracy: 93.8%
```
Now we can see that the accuracy slowly improves over time, but nothing seams to get past ~95%, either we're getting stuck in some local minimum or our neural net is just too simple to get past this point.

Increasing the nodes in our hidden layer from 10 to 128 increases the computation needed significantly, but we also get a sizable improvement to our accuracy:

```tex
Iteration [140]
Accuracy: 97.6%
Iteration [150]
Accuracy: 97.6%
```
Although we again hit a ceiling and can't seam to get past ~98%. We can try to add more layers to our NN and see if that can help us. Instead of one 128 long layer, we'll assign two layers with the same nodes per layer:

```tex
Accuracy: 97.0%
Iteration [40]
Accuracy: 97.0%
Iteration [50]
Accuracy: 97.0%
```

Computation now takes significantly longer but we still can't seam to cross the 98& barrier, even when re-training the NN multiple times, we could try to add momentum to our design but it's doubtful it will yield the needed boost, changing our activation function to ReLu might be a better approach.

## SGD w/ Changing learning rate + momentum
Didn't do this ._. don't think it's needed for the assignment and I'm happy where I am thus far.