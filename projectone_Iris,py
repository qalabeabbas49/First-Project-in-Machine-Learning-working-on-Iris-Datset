#Iris Dataset Step by Step


# 1. Prepare Problem
# a) Load Libraries 
# Load all the required modules and libraries that will be used for our problem.
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# b) Load Dataset
# Load tha dataset from file/url and this is the place to reduce the sample of dataset specially if it's too large to work with.
# We can always scale up the well performing models later.

filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)

# 2. Summarize Data
# This step is to learn more about the dataset which will help us decide which algorithms to use with this data. 
# a) Descriptive Statistics
#shape
print(dataset.shape)
#head
print (dataset.head())
#description
print(dataset.describe())
#class distribution
print(dataset.groupby('class').size())

# b) Data Visualizations
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
#histogram
dataset.hist()
pyplot.show()
#scatter plot matrix - to spot relationships between input variables
scatter_matrix(dataset)
pyplot.show()

# 3. Prepare Data
# a) Data Cleaning - Removing duplicates, dealing with the missing values
# b) Feature Selection
# c) Data Transforms - Scaling/standarizaion of data
#This step is not needed here as the data is already clean and ready to use


# 4. Evaluate Algorithm
# a) Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# b) Test options and evaluate metric
#10-fold cross validation is used to estimate accuracy om unseen data. This split the data into 10 parts, the model will train
# on 9 parts and test on 1. 
#This step in combined in the next step
# c) Spot Check Algorithms
models =[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# d) Compare Algorithms

fig = pyplot.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles
# since the data is very simple and basic and we have already achieved an accuracy of 99% with svm, we don't need this step here.


# 6. Finalize Model
# a) Predictions on validation dataset
svm =SVC(gamma='auto')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


