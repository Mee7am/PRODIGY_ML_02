import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as born
from sklearn.cluster import KMeans

#Importing data
data = pd.read_csv("Mall_Customers.csv")

#Choosing the Annual Income Column and Spending score column
lst1 = data.iloc[:,[3,4]].values

#Choosing the number of clusters
lst2 = []
for i in range(1,11):
    kmean = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmean.fit(lst1)
    lst2.append(kmean.inertia_)

#Code for plotting elbow graph to identify optimum number of clusters
"""
born.set()
plot.plot(range(1,11), lst2)
plot.title("Elbow Point Graph")
plot.xlabel("No. of Clusters")
plot.ylabel("Within Clusters Sum of Squares")
plot.show()

#Optimum clusters were 5
"""

#Training k-mean clustering method
kmean = KMeans(n_clusters=5, init="k-means++", random_state=0)

#Returning a label for each data point based on their cluster
lst3 = kmean.fit_predict(lst1)


"""
print(lst3)
#The output for the above is as followed
# [3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3
#  4 3 4 3 4 3 0 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 1 2 1 0 1 2 1 2 1 0 1 2 1 2 1 2 1 2 1 0 1 2 1 2 1
#  2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2
#  1 2 1 2 1 2 1 2 1 2 1 2 1 2 1]

which means that clusters are 5
"""

#For visualizing the clusters the code is written as below

plot.figure(figsize=(8,8))
plot.scatter(lst1[lst3==0,0], lst1[lst3 == 0,1], s=50, c="red", label="Cluster-1" )
plot.scatter(lst1[lst3==1,0], lst1[lst3 == 1,1], s=50, c="yellow", label="Cluster-2" )
plot.scatter(lst1[lst3==2,0], lst1[lst3 == 2,1], s=50, c="green", label="Cluster-3" )
plot.scatter(lst1[lst3==3,0], lst1[lst3 == 3,1], s=50, c="blue", label="Cluster-4" )
plot.scatter(lst1[lst3==4,0], lst1[lst3 == 4,1], s=50, c="purple", label="Cluster-5" )

#Marking the centroid for the clusters and genertaing plot
plot.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:,1], s=150, c="black", marker="x", label="Centroid")
plot.title("K-Means Clustering")    
plot.xlabel("Annual Income")
plot.ylabel("Spending values")
plot.show()