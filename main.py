from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

def load_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=names)

    dataset.plot(kind='bar', subplots=True, layout=(2,2), sharex=False, sharey=False)
    dataset.hist(color="red")
    pyplot.show()

def main():
    load_dataset()


if __name__ == "__main__":
    main()
