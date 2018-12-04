# Links
- The [code](https://github.com/amueller/introduction_to_ml_with_python)
- a home made lib called [mglearn](https://github.com/amueller/introduction_to_ml_with_python/tree/master/mglearn)
# Intro
## 2 types of ML algorithms:
- **supervised**
  
  has known input & output data, ex:
  - recognize handwriting zipcode
  - medical diagnose about tumor
  - fraudulent activities of credit cards

- **unsupervised**
    
    only has known input data
    - identify topics of blog posts
    - segment customer groups by preferences
    - abnormal access patterns to website

  *The main difference between 2 seems like supervised learning has only one output, and the unsupervised one has unknown number of output?* => **not really, s has known outputs, which means no input can create new, unknown outputs, but u will generate unknown outputs depends on our features.**

## The data
If we see data as a table, then
- each *row* or *entity* is known as **sample** (or **data point**)
- each *column* or *property* to describe the entity is known as **feature**.

## The task
If the dataset we have has no useful informations related with the task we want to archive, then no algorithm can help.

**The keys:**
- know the question, and decide if
  - the data is **capable** of answering that question
  - we have **enough** data
- shape the question as a ml problem
- Which features inside data to **extract**, and how to do the prediction
- How to measure the **accuracy** of result
- How to use the ml solution to interact with other applications & business demands.


## What we need as tools
- [Anaconda](https://www.anaconda.com/download/#macos)

- or inside any chosen directory
  1. `python3 -m venv name-of-env-folder`
  2. `source name-of-env-folder/bin/activate`
  3. `pip install numpy scipy matplotlib ipython jupyter scikit-learn pandas`
  4. open notebook with `jupyter notebook`, for more running options, see the [doc](https://jupyter.readthedocs.io/en/latest/running.html#running)
  5. create new notebook from the opened web interface, and start typing...

#### A little breakdown for tools
- `ipython` with `jupyter`: to run an interactive webpage that can display directly results & charts
- `numpy`: multidimensional array, math functions like linear algebra or Fourier transform, random number generator, etc. N.B. all entities inside an array must be the same type.
- `scipy`: collection of scientific computing functions. Most important one is `scipy.sparse` to create *sparse matrices*. A cookbook about scipy [here](http://www.scipy-lectures.org/).
- `matplotlib`: an OO lib to create visualizations
- `pandas`: lib for data analysis, build around a  data structure called *DataFrame*, which can be simply put as a data table, and provide SQL-like queries & joins for tables. 
- `scikit-learn`: lib that offers tons of predifined ML methods...

## Roll the dice (Task 1)
A classification (a supervised one) of iris (I prefer dogs actually...)
#### some tech terms:
- the possible outputs are called *classes*
- for every data point (an iris image in our case), the species it belongs called its *label*

### Start with the [principle](#the-task)
- know your data first
  ```python
  print('data type', type(dataset))

  # if dataset is a dict-like structure
  print('keys of data', dataset.keys())

  for (k, v) in dataset.items():
    print(k,v)

  # if dataset is a list-like structure
  print('first 5 data', dataset[0:5])
  ```

- decide how to measure success

  split dataset to *training set* & *test set*: 
`scikit-learn` has a build-in function called `train_test_split` to **shuffle** the dataset, which automatically takes 75% of the data as training set, and 25% as the test set. This partition can be a good default for most tasks.

  The `train_test_split` function takes an argument called `random_state`, to give a **fixed seed** to pseudorandom number generator, that makes sure we always get the *same output* whenever we run this function, we can use this when using randomrize procedures but want consistant output:
  ```python
  from sklearn.model_selection import train_test_split

  # it doesn't matter which number you use for random_state, just be consistent.
  X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state = 42)
  ```
  Another param `stratify` of `train_test_split` makes a huge difference, it means split the dataset according to the **proportion** of the specified column. Ex: `stratify=y`, if `y` has 75% `0`s, and 25% `1`s, then it will ensure that both the training set & the test set will have 75% `0`s, 25% `1`s.

  **conventions**: we use capital `X` to denote dataset, as it's a 2-dimentional array (**matrix**), and use lowercase `y` to denote target, as it's a 1-dimensional array (**vector**).

  **TODO**: How the `train_test_split` knows how to match the `dataset` with `target` ? It means we have to **sort** our data first, to make the columns of data & target have same index, and a perfect 1-1 relationship ?

- deeper look at data by vis
  - One common way is using a *scatter plot*, but since it can deal with only 2 features (one along x-axis, another along y-axis), it's difficult to vis multiple features.
  - Or if we have more features, we can try *pair plot*, which look at *all* possible pairs of features. **N.B.**: the pair plot does not show interactivities of all features, so we may lost some insight by visualize this way.

  **TODO**: that's the same for both scatter plot & pair plot no?

- build a simple k-nearest neighbors classification model, which means we decide a new data point by *k* nearest points in the training dataset, so the model building algo needs only remember all training dataset 🎉 :
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  # we use only single neighbor this time
  # this method returns an object that contains the algo to build model, and the algo to predicte new data points
  knn = KNeighborsClassifier(n_neighbors=1)
  # the fit method inside knn returns the knn object itself, with modifications in place
  # feed it with our input training set, and corresponding output training set
  knn.fit(X_train, y_train)
  ```

- evaluate the model with test set
Measure how the model works using *accuracy*, that means, the percentage of right predicted iris
  `knn.score(X_test, y_test)`

## sum up
Just use `fit`, `predict` & `score` methods from sklearn and normally you are good to go. 😂
- `fit` need to pass in the `X_train` & `y_train`
- `predict` takes test set `X_test`, and output the predictions (normally an np array)
- `score` simply takes `X_test` & `y_test`, internally it use `predict` on test set, and compare the result with test output.

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

  

















