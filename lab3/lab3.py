import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import statistics
from sklearn import preprocessing
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
import random

f = open("data.txt")
lines = f.readlines()
numberOfAttr = 10
data = []
label = []

dataG = []
dataH = []
for i in range(len(lines)):
    line = lines[i].split(',')
    data.append([float(x) for x in line[0:len(line)-1]])
    curLabel = line[len(line)-1]
    if curLabel[-1] == '\n':
        curLabel = curLabel[0]


    if curLabel == 'g':
        data[-1].append(0)
    else :
        data[-1].append(1)

    if curLabel == 'g':
        dataG.append(data[-1])
    else:
        dataH.append(data[-1])

numberOfRowsToDelete = 12332 - 6688
for i in range(numberOfRowsToDelete):
    j = random.randint(0,len(dataG)-1)
    data.remove(dataG[j])
    dataG.pop(j)

for i in range(len(data)):
    label.append(data[i][-1])
    data[i].pop()

# min_max_scaler
min_max_scaler = preprocessing.MinMaxScaler()
data_min_max_scaled = min_max_scaler.fit_transform(data)


# correlation Matrix
correlation_matrix = np.corrcoef(x=data_min_max_scaled, rowvar=False)

plt.title('Correlation Matrix')
plt.imshow(correlation_matrix)
plt.show()


dataG = []
dataH = []
for i in range(len(data_min_max_scaled)):
    if label[i] == 0:
        dataG.append(data_min_max_scaled[i])
    else:
        dataH.append(data_min_max_scaled[i])


# Scater plot
for i in range(numberOfAttr):
    col1H = [row[i] for row in dataH]
    col1G = [row[i] for row in dataG]
    for j in range(i+1,numberOfAttr):
        col2H = [row[j] for row in dataH]
        col2G = [row[j] for row in dataG]
        plt.scatter(col1H, col2H)
        plt.scatter(col1G, col2G)
        plt.title('Scatter plot')
        plt.xlabel(i)
        plt.ylabel(j)
        plt.show()


# select k best
selector = SelectKBest(chi2, k=9)
data_best = selector.fit_transform(data_min_max_scaled, label)
# Data splitting
X_train, X_test, y_train, y_test = train_test_split(data_best, label, test_size=0.3)



from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(name, clf, cleanClf):
    print(name)
    print('Acc =', accuracy_score(y_test , clf.predict(X_test)))
    print('Confusion Matrix')
    print(confusion_matrix(y_test, clf.predict(X_test)))
    print(classification_report(y_test, clf.predict(X_test)))
    print('Cross-validation mean error =', cross_val_score(cleanClf, X, Y, cv=5).mean())
    print()

if __name__ == '__main__':
    X, Y = X_train, y_train

    # Decision Tree
    clf = tree.DecisionTreeClassifier().fit(X, Y)
    evaluate('Decision Tree', clf, tree.DecisionTreeClassifier())

    # Naive Bayes
    clf = GaussianNB().fit(X, Y)
    evaluate('Naive Bayes', clf, GaussianNB())

    # Random Forest
    for i in range(8,13):
        clf = RandomForestRegressor(n_estimators = i).fit(X, Y)
        print('Random Forest n_estimators =', i);
        print('Acc =', accuracy_score(y_test, [round(x) for x in clf.predict(X_test)]))
        print('Confusion Matrix')
        print(confusion_matrix(y_test, [round(x) for x in clf.predict(X_test)]))
        print(classification_report(y_test, [round(x) for x in clf.predict(X_test)]))
        print('Cross-validation mean error =', cross_val_score(RandomForestRegressor(n_estimators = i), X, Y, cv=5).mean())
        print()

    # Adaboost
    for i in range(8,13):
        clf = AdaBoostClassifier(n_estimators=i).fit(X, Y)
        evaluate('Adaboost n_estimators = ' + str(i), clf, AdaBoostClassifier(n_estimators=i))

    # KNN
    for i in range(8,13):
        clf = KNeighborsClassifier(n_neighbors=i).fit(X, Y)
        evaluate('KNN K = ' + str(i), clf, KNeighborsClassifier(n_neighbors=i))
