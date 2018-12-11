# k-Nearest Neighbors (k-NN for short)
  
Simplest model, which the *build model algo* is just **remembering all the training set**. And the *predict algo* is trying to find the nearest `k` elements of the test data point, and do a `voting` among those `k` elements, to decide the prediction.

To analyse the k-NN classification model with different `k` values, we can draw a plot that colors different classes to see the **decision boundary**.

For k-NN, the higher the `n_neighbors`, the simpler (smoother) the model, and it's important to find a sweet point.

Considering if you use `1` for n_neighbors, then the training set will always has the highest accuracy, but it's a `overfitting` as we discussed, it might not perform well on unseen data. 

On the other hand, if we go extreme, to include *all* data points in training set, then every new data points will have exact the same neighbors (the whole training data points), thus the exact same output, that's clearly an `underfitting`.

**Different k-NN types**: 
  - For *classification*, after finding the k nearest sample, take the *most frequent label* of their labels. 
  - For *regression*, we can take the *mean or median of the k neighbors*, or we can solve a linear regression problem on the neighbors.

**2 keys params of k-NN**:
  - number of neighbors
  - the algo to calculate distance (by default we use *Euclidean distance*)

 **N.B.**:
  - the prediction of k-NN is *slow* when data set is large, it's important to preprecess the data before
  - k-NN works badly on data set with lots (>100) of features, and especially badly on *sparse dataset* ( most of features are `0`).

## Next
[Linear model as supervised models](/04-supervised-ml-linear.md)