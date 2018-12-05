# Supervised ML, seriously
The supervised learning normally requires human effort to build the training set, specially for the training output. But can be automated afterwards once data are ready.

## 2 major types of supervised ml
- **Classifiction**
  
  with goal to predict a *class label* , that should be chosen from a predefined list. Can be further break down to 2 sub types:
  - binary classification (yes/no question), **EX**: spam mail.
  - multiple classes classification, **EX**: the above iris one, or like predict which language a website is in.

- **Regression**

  with goal to predict a *continous (floating point) number*.

  **EX** : predict a person's annual income based on their age, education, location, the result is an *amount*, which can be any number within a range. Or predict the yield (产量) of a corn field based on last year's yield & weather, ect.

**The difference between those 2 is the *continuity* of the output, if one should have precise output, without something in between, then it should be a classification, otherwise, like incomes, 3999 & 4001 doesn't make much difference.**

## About models
The key metric to a model that hits the sweetpoint: **generalization**, the ability to make accurate prediction on **unseen data**, that's the whole point of ml models. 

Normaly, simple model generalize better to new data. If we create a *too complex* model based on existing dataset, we risk having:

- **overfitting**: which means you fit a model too closely to the *particularities* of training set -- it's not general enough. => **Normally shows high accuracy on training set, and low accuracy on test set**.

But if we have a model *too simple*, like 'anyone with a house will buy a boat', then we will have :

- **underfitting**: that ignored too much varaibilities of the dataset, which will not perform well even on training set. => **Normally shows low & very close accuracy on both training set and test set**.

💩 So our model cannot be neither too simple nor too complex, we need to find a sweet point between those 2, and the tradeoff of **overfitting** and **underfitting** can be shown as below:
![trade-off-overfitting-underfitting](./assets/tradoff.png) 

**model complexity is relative with data size**

The more data we have, the more complex can be the model, without overfitting. (样本越大多样性越强，所以更不容易被单一化).

## List of common supervised ml models

#### Side note about some `python` & `NumPy` functions:
- `np.bincount` can be used to count the *occurence* of a value inside an array of `int`, manual [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html). 

  Put it in human readable language: feed it an array, `bincount` will
  - first take the *largest* int, to decide the output array size. 
  - then count how many times each value appeared
  - you can add a second param known as `weight`, which should be exactly the same shape of input array

  Ex:
  ```python
  x = np.array([9,9,8])
  # the output array size will be 10 as the largest int is 9
  print('count of each value:', np.bincount(x))
  # [0 0 0 0 0 0 0 0 1 2]
  # 8 appeared once, 9 appeared twice, all other values from 0-7 appeared 0 time. 

  # using weight
  w = np.array([0.5, 0.3, 1])
  print('count of value with weight:', np.bincount(x, w))
  # [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.8]
  ```

- `zip` is a python BIF used to zip 2 arrays together, returns an *iterator of tuples*, where the `i`-th tuple corresponds with `i`-th elements in each array (the longer array is truncated).

#### Preparing the data
- We can include some derived features from the *products* of original features, it's called **feature engineering**. We can use the combination of `k` elements from a set of `n` elements, called *`n` choose `k`*.

### Various types of supervised ml models
- #### k-Nearest Neighbors (k-NN for short)
  
  Simplest model, which the *build model algo* is just remembering all the training set. And the *predict algo* is trying to find the nearest `k` elements of the test data point, and do a `voting` among those `k` elements, to decide the prediction.

  To analyse the k-NN classification model with different `k` values, we can draw a plot that colors different classes to see the **decision boundary**.

  For k-NN, the higher the `n_neighbors`, the simpler (smoother) the model, and it's important to find a sweet point.

  Considering if you use `1` for n_neighbors, then the training set will always has the highest accuracy, but it's a `overfitting` as we discussed, it might not perform well on unseen data. On the other hand, if we go extreme, to include *all* data points in training set, then every new data points will have exact the same neighbors (the whole training data points), thus the exact same output, that's clearly an `underfitting`.

  **Different k-NN types**: 
  - For *classification*, after finding the k nearest sample, take the *most frequent label* of their labels. 
  - For *regression*, we can take the *mean or median of the k neighbors*, or we can solve a linear regression problem on the neighbors.

  2 keys params of k-NN:
  - number of neighbors
  - the algo to calculate distance (by default we use *Euclidean distance*)

  **N.B.**:
  - the prediction of k-NN is slow when data set is large, it's important to preprecess the data before
  - k-NN works badly on data set with lots (>100) of features, and especially badly on *sparse dataset* ( most of features are `0`).

- #### Linear Models

  Use a *linear function* to the input features, to get the prediction. The most common one is linear regression =>`y = W * X + b`.

  The *linear* assumption seems too ideal to fit into features, but as long as `NO of features > NO of data points`, the target (prediction) `y` can be perfectly modeled as a *linear function*.

  - Regression models
    - *Linear regression (ordinary least squares, OLS for short, 最小方差)*: 
    
      find the `w` & `b` by minimize the **mean squared error** between prediction & target. 
    
      It has no params when creating models (unlike the `n_neighbors` param for k-NN), which makes it simpler, but hard to control the complexity of model. The returned object by `LinearRegression` from `sklearn.linear_model` contains `coef_` as `w`, and `intercept_` as `b`. Detailed doc [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

      *N.B.*: the `_` after means it's derived from the training set, to distinguish with the user set params.

      Linear regression is simple to use & understand, but if it doesn't perform well on some dataset, as we cannot control the complexity by providing params, we should try other models.

    - *Ridge regression*

      Mostly the same as linear regression, but we want the `coefficient(w)` to meet extra contraint, and be close to `0`. The additional contraint is called **regularization**. And we use *L2 regularization* for Ridge regression to prevent *overfitting*.

      We can adjust the contraint of `w` by param `alpha` from the `Ridge` model in `sklearn`, by default it takes `alpha=1.0`, the higher the number, the more `w` is near `0`. That might make the algo perform worse on training set, but more generalized. if we put `alpha=0`, the constraint is none and the **Ridge regression** performs the same as **Linear regression**.

      To have an idea about how different `alpha` values influence the *coefficient*, we can
      - plot the `coef_` generated by different `alpha`s, to see the magnitude of `coef_`s (the closer to 0, the more generalized)
      - or the **learning curve**, where we show the model performance as `f(dataset_size)`  

  