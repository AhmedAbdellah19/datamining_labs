from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loading The Iris Dataset
iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names
attributes = iris.feature_names

# Cosine Similarity Matrix
similarities = cosine_similarity(data)
print('Cosine Similarity Matrix:\n', similarities)

plt.title('Similarity Matrix')
plt.imshow(similarities)
plt.show()

# Visualization
classes = [[x for i,x in enumerate(data) if target[i] == 0],
		   [x for i,x in enumerate(data) if target[i] == 1],
		   [x for i,x in enumerate(data) if target[i] == 2]]

# X data for each class alone
for i in range(len(classes)):
	fig, ax = plt.subplots()
	plt.title('X plot for class ' + target_names[i].upper())
	for j in range(len(attributes)):
		ax.plot([row[j] for row in classes[i]], label=attributes[j])
	ax.legend()
	plt.show()

# Histogram for each class
for i in range(len(classes)):
	for j in range(len(attributes)):
		plt.title('Histogram for attribute ' + attributes[j] + ' of class ' + target_names[i].upper())
		plt.hist([a[j] for a in classes[i]])
		plt.show()

# 2D Scatter Plot
for i in range(len(attributes)):
	for j in range(i+1, len(attributes)):
		fig, ax = plt.subplots()
		plt.title('2D Scatter Plot')
		plt.xlabel(attributes[i])
		plt.ylabel(attributes[j])
		for c in range(len(classes)):
			x = [a[i] for a in classes[c]]
			y = [a[j] for a in classes[c]]
			ax.scatter(x, y, label=target_names[c])
		ax.legend()
		plt.show()

# 3D Scatter Plot
for i in range(len(attributes)):
	for j in range(i+1, len(attributes)):
		for k in range(j+1, len(attributes)):
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			plt.title('3D Scatter Plot')
			ax.set_xlabel(attributes[i])
			ax.set_ylabel(attributes[j])
			ax.set_zlabel(attributes[k])
			for c in range(len(classes)):
				x = [a[i] for a in classes[c]]
				y = [a[j] for a in classes[c]]
				z = [a[k] for a in classes[c]]
				ax.scatter(x, y, z, label=target_names[c])
			ax.legend()
			plt.show()