from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
def load_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

    dataset.plot(kind='bar', subplots=True, layout=(2,2), sharex=False, sharey=False)
    dataset.hist(color="red")
    pyplot.show()


def split_dataset(dataset):
    print(dataset.describe)
    learn_array = dataset.values
    X = learn_array[:0:4]
    Y = learn_array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X,y,test_size=0.20, random_state=1)
def main():
    load_dataset()
    split_dataset()


if __name__ == "__main__":
    main()
