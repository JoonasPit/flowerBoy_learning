from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LinDiA
from sklearn.tree import DecisionTreeClassifier as DecTree
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score as c_v_l


# ALlter

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


def load_dataset(dataset):

   # dataset.plot(kind='bar', subplots=True, layout=(2,2), sharex=False, sharey=False)
   # dataset.hist(color="red")
  #  pyplot.show()

 #   print(dataset.describe)
    learn_array = dataset.values
    X = learn_array[:, 0:4]
    Y = learn_array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# add to array of models name, model
    models = []
    models.append(('LR', LogReg(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinDiA()))
    models.append(('CART', DecTree()))
    models.append(('NB', GNB()))
    models.append(('SVM', SVC(gamma='auto')))

    results = []
    names = []
    # choose a validation method, run through c_v_l function with model being selected from models array
   # Get put in to results array.
   # boxplot out results from results array
   # Chose best algorithm based off result accuracy
    for name, model in models:
        fold = StratifiedKFold(n_splits=10, random_state=1)
        validated_results = c_v_l(model, X_train, Y_train, cv=fold, scoring='accuracy')
        results.append(validated_results)
        names.append(name)
        print('%s: %f (%f)' % (name, validated_results.mean(), validated_results.std()))


    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()
def main():
    load_dataset(dataset)


if __name__ == "__main__":
    main()
