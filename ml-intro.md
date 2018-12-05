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

