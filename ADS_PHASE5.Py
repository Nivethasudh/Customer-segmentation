Import numpy as np

Import pandas as pd

Import matplotlib.pyplot as plt

Import seaborn as sns
Df = pd.read_csv(“../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv”)
Df.head()
Df.drop([“CustomerID”], axis = 1, inplace=True)
Plt.figure(figsize=(10,6))
Plt.title(“Ages Frequency”)

Sns.axes_style(“dark”)

Sns.violinplot(y=df[“Age”])

Plt.show()
Age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]

Age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]

Age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]

Age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]

Age55above = df.Age[df.Age >= 56]
X = [“18-25”,”26-35”,”36-45”,”46-55”,”55+”]

Y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

Plt.figure(figsize=(15,6))

Sns.barplot(x=x, y=y, palette=”rocket”)

Plt.title(“Number of Customer and Ages”)

Plt.xlabel(“Age”)

Plt.ylabel(“Number of Customer”)

Plt.show()
Ss1_20 = df[“Spending Score (1-100)”][(df[“Spending Score (1-100)”] >= 1) & (df[“Spending Score (1-100)”] <= 20)]

Ss21_40 = df[“Spending Score (1-100)”][(df[“Spending Score (1-100)”] >= 21) & (df[“Spending Score (1-100)”] <= 40)]

Ss41_60 = df[“Spending Score (1-100)”][(df[“Spending Score (1-100)”] >= 41) & (df[“Spending Score (1-100)”] <= 60)]

Ss61_80 = df[“Spending Score (1-100)”][(df[“Spending Score (1-100)”] >= 61) & (df[“Spending Score (1-100)”] <= 80)]

Ss81_100 = df[“Spending Score (1-100)”][(df[“Spending Score (1-100)”] >= 81) & (df[“Spending Score (1-100)”] <= 100)]

Ssx = [“1-20”, “21-40”, “41-60”, “61-80”, “81-100”]

Ssy = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]

Plt.figure(figsize=(15,6))

Sns.barplot(x=ssx, y=ssy, palette=”nipy_spectral_r”)

Plt.title(“Spending cores”)

Plt.xlabel(“Score”)

Plt.ylabel(“Number of Customer Having the Score”)

Plt.show()

Ai0_30 = df[“Annual Income (k$)”][(df[“Annual Income (k$)”] >= 0) & (df[“Annual Income (k$)”] <= 30)]

Ai31_60 = df[“Annual Income (k$)”][(df[“Annual Income (k$)”] >= 31) & (df[“Annual Income (k$)”] <= 60)]

Ai61_90 = df[“Annual Income (k$)”][(df[“Annual Income (k$)”] >= 61) & (df[“Annual Income (k$)”] <= 90)]

Ai91_120 = df[“Annual Income (k$)”][(df[“Annual Income (k$)”] >= 91) & (df[“Annual Income (k$)”] <= 120)]

Ai121_150 = df[“Annual Income (k$)”][(df[“Annual Income (k$)”] >= 121) & (df[“Annual Income (k$)”] <= 150)]
Aix = [“$ 0 – 30,000”, “$ 30,001 – 60,000”, “$ 60,001 – 90,000”, “$ 90,001 – 120,000”, “$ 120,001 – 150,000”]

Aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]
Plt.figure(figsize=(15,6))

Sns.barplot(x=aix, y=aiy, palette=”Set2”)

Plt.title(“Annual Incomes”)

Plt.xlabel(“Income”)

Plt.ylabel(“Number of Customer”)

Plt.show()
From sklearn.cluster import KMeans

Wcss = []

For k in range(1,11):

    Kmeans = KMeans(n_clusters=k, init=”k-means++”)

    Kmeans.fit(df.iloc[:,1:])

    Wcss.append(kmeans.inertia_)

Plt.figure(figsize=(12,6))    

Plt.grid()

Plt.plot(range(1,11),wcss, linewidth=2, color=”red”, marker =”8”)

Plt.xlabel(“K Value”)

Plt.xticks(np.arange(1,11,1))

Plt.ylabel(“WCSS”)

Plt.show() 

Km = KMeans(n_clusters=5)

Clusters = km.fit_predict(df.iloc[:,1:])

Df[“label”] = clusters

From mpl_toolkits.mplot3d import Axes3D

Import matplotlib.pyplot as plt

Import numpy as np

Import pandas as pd
Fig = plt.figure(figsize=(20,10))

Ax = fig.add_subplot(111, projection=’3d’)

Ax.scatter(df.Age[df.label == 0], df[“Annual Income (k$)”][df.label == 0], df[“Spending Score (1-100)”][df.label == 0], c=’blue’, s=60)

Ax.scatter(df.Age[df.label == 1], df[“Annual Income (k$)”][df.label == 1], df[“Spending Score (1-100)”][df.label == 1], c=’red’, s=60)

Ax.scatter(df.Age[df.label == 2], df[“Annual Income (k$)”][df.label == 2], df[“Spending Score (1-100)”][df.label == 2], c=’green’, s=60)

Ax.scatter(df.Age[df.label == 3], df[“Annual Income (k$)”][df.label == 3], df[“Spending Score (1-100)”][df.label == 3], c=’orange’, s=60)

Ax.scatter(df.Age[df.label == 4], df[“Annual Income (k$)”][df.label == 4], df[“Spending Score (1-100)”][df.label == 4], c=’purple’, s=60)

Ax.view_init(30, 185)

Plt.xlabel(“Age”)

Plt.ylabel(“Annual Income (k$)”)

Ax.set_zlabel(‘Spending Score (1-100)’)

Plt.show()
