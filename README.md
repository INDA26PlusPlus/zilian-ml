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

## SGD w/ Changing learning rate + momentum
