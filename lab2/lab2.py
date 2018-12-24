import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import statistics
from sklearn import preprocessing
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

f = open("segmentation.data.txt")
attributes = f.readline().split(',')
attributes[-1] = attributes[-1][:-1]
print('Attributes: ', attributes)
f.readline()
data = []
lines = f.readlines()
classes = []
for line in lines:
    line = line[:-1].split(',')
    classes.append(line[0])
    data.append([float(x) for x in line[1:]])

# checking and preproccessing data
print("Number of readings: ", len(data))
print("Number of attributes: " , len(attributes))
classNames = list(set(classes))
print("Number of classes: ", len(classNames))


# data visualization
# plt.title('Data visualization')
# plt.plot(data, linestyle='none', marker='o')
# plt.show()


# classes tuples
classTuples = []
for className in classNames:
    classTuples.append([data[i] for i in range(len(data)) if classes[i] == className])

# for i in range(len(classTuples)):
#     plt.title(classNames[i])
#     plt.plot(classTuples[i], linestyle='none', marker='o')
#     plt.show()


for i in range(len(classTuples)):
    for binsNumber in [5, 10, 12]:
        plt.title(classNames[i] + " " + str(binsNumber))
        for j in range(len(attributes)):
            col = [row[j] for row in classTuples[i]]
            plt.hist(col, bins=binsNumber)
        plt.show()

# pearson correlation Matrix
pearsonMatrix = [[0 for i in range(len(attributes))] for j in range(len(attributes))]
for i in range(len(attributes)):
    col1 = [row[i] for row in data]
    for j in range(i+1,len(attributes)):
        col2 = [row[j] for row in data]
        pearsonMatrix[i][j] = pearsonMatrix[j][i] = pearsonr(col1, col2)[0]

plt.title('pearson correlation Matrix')
plt.imshow(pearsonMatrix)
plt.show()

# covariance Matrix
for i in range(len(attributes)):
    col1 = [row[i] for row in data]
    for j in range(i+1,len(attributes)):
        col2 = [row[j] for row in data]
        covMatrix = np.cov(col1, col2)
        print('Covariance Matrix of ' + attributes[i] + ' and ' + attributes[j], covMatrix)


# Histogram for data to compare
plt.title("Histogram for data")
plt.hist(data)
plt.show()

# min_max_scaler
min_max_scaler = preprocessing.MinMaxScaler()
data_min_max_scaled = min_max_scaler.fit_transform(data)
plt.title("Histogram for min max scaled data")
plt.hist(data_min_max_scaled)
plt.show()

# z-score (with removing 3rd column)
data_zscore_scaled = stats.zscore([row[0:2] + row[3:] for row in data])
plt.title("Histogram for z score scaled data")
plt.hist(data_zscore_scaled)
plt.show()

# PCA 15 attribute
pca = PCA(n_components=15)
data_pca_15 = pca.fit_transform(data_zscore_scaled)

# pearson correlation Matrix
pearsonMatrix = [[0 for i in range(15)] for j in range(15)]
for i in range(15):
    col1 = [row[i] for row in data_pca_15]
    for j in range(i+1,15):
        col2 = [row[j] for row in data_pca_15]
        pearsonMatrix[i][j] = pearsonMatrix[j][i] = pearsonr(col1, col2)[0]

plt.title('Pearson Correlation Matrix PCA 15')
plt.imshow(pearsonMatrix)
plt.show()
print('pca.explained_variance_ratio_ = ', pca.explained_variance_ratio_)

# PCA 12 attribute
pca = PCA(n_components=12)
data_pca_12 = pca.fit_transform(data_zscore_scaled)

# pearson correlation Matrix
pearsonMatrix = [[0 for i in range(12)] for j in range(12)]
for i in range(12):
    col1 = [row[i] for row in data_pca_12]
    for j in range(i+1,12):
        col2 = [row[j] for row in data_pca_12]
        pearsonMatrix[i][j] = pearsonMatrix[j][i] = pearsonr(col1, col2)[0]

plt.title('Pearson Correlation Matrix PCA 12')
plt.imshow(pearsonMatrix)
plt.show()
print('pca.explained_variance_ratio_ = ', pca.explained_variance_ratio_)


# select k best
classNameToIndex = {}
for i in range(len(classNames)):
    classNameToIndex[classNames[i]] = i

y = [classNameToIndex[i] for i in classes]
selector = SelectKBest(chi2, k=15)
data_best_15 = selector.fit_transform(data_min_max_scaled, y)

# data visualization
plt.title('SelectKBest 15')
plt.plot(data_best_15, linestyle='none', marker='o')
plt.show()

# pearson correlation Matrix
pearsonMatrix = [[0 for i in range(15)] for j in range(15)]
for i in range(15):
    col1 = [row[i] for row in data_best_15]
    for j in range(i+1,15):
        col2 = [row[j] for row in data_best_15]
        pearsonMatrix[i][j] = pearsonMatrix[j][i] = pearsonr(col1, col2)[0]

plt.title('Pearson Correlation Matrix SelectKBest 15')
plt.imshow(pearsonMatrix)
plt.show()


# select k best
selector = SelectKBest(chi2, k=12)
data_best_12 = selector.fit_transform(data_min_max_scaled, y)

# data visualization
plt.title('SelectKBest 12')
plt.plot(data_best_12, linestyle='none', marker='o')
plt.show()

# pearson correlation Matrix
pearsonMatrix = [[0 for i in range(12)] for j in range(12)]
for i in range(12):
    col1 = [row[i] for row in data_best_12]
    for j in range(i+1,12):
        col2 = [row[j] for row in data_best_12]
        pearsonMatrix[i][j] = pearsonMatrix[j][i] = pearsonr(col1, col2)[0]

plt.title('Pearson Correlation Matrix SelectKBest 12')
plt.imshow(pearsonMatrix)
plt.show()