# Academic Projects for CSE-6363 Machine Learning

## kNN - Classifier
Iris data set is used in .csv format.  
Downloaded from [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)  
It includes three iris species with 50 samples each as well as some properties about each flower.  
One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.  
The columns in this dataset are:  

1. Id
2. SepalLengthCm
3. SepalWidthCm
4. PetalLengthCm
5. PetalWidthCm
6. Species

The code is contained in [knn-classifier.py](knn-classifier/knn-classifier.py)  
It contains 6 functions -
1. loadDataset  
    Loads the dataset from a .csv file. First instance is assumed to contain feature labels, so it is skipped.
2. getDistance  
    Returns euclidean distance between two vectors
3. getNeighbours  
    Returns k nearest neighbours to a test instance
4. predictClass  
    Returns the most likely class from all the neighbours
5. getAccuracy  
    Tests the predictions against the entire test dataset. Accuracy is printed in percentage
6. myknnclassify  
    Required function mentioned in the problem  

Value of k: 12  
Training and test data are created by randomly splitting data in 66:34 ratio.  
Classifier accuracy generally >**95**%

<hr>

## kNN - Regressor
Fertility dataset is used in .csv format.  
Downloaded from [Fertitlity Dataset](https://archive.ics.uci.edu/ml/datasets/Fertility)  
100 volunteers provide a semen sample analyzed according to the WHO 2010 criteria.  
Sperm concentration are related to socio-demographic data, environmental factors, health status, and life habits  
Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1)  
Age at the time of analysis. 18-36 (0, 1)  
Childish diseases (ie , chicken pox, measles, mumps, polio)	1) yes, 2) no. (0, 1)  
Accident or serious trauma 1) yes, 2) no. (0, 1)  
Surgical intervention 1) yes, 2) no. (0, 1)  
High fevers in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1)  
Frequency of alcohol consumption 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1)  
Smoking habit 1) never, 2) occasional 3) daily. (-1, 0, 1)  
Number of hours spent sitting per day ene-16	(0, 1)  
Output: Diagnosis	normal (N), altered (O)  

The code is contained in [knn-regressor.py](knn-regressor/knn-regressor.py)  
It contains 6 functions -  
1. loadDataset  
    Loads the dataset from a .csv file. First instance is assumed to contain feature labels, so it is skipped.
2. getDistance  
    Returns euclidean distance between two vectors
3. getNeighbours  
    Returns k nearest neighbours to a test instance
4. calculateValue  
    Returns the most likely class from all the neighbours
5. getAccuracy  
    Tests the predictions against the entire test dataset. Accuracy is printed in percentage
6. myknnregress  
    Required function mentioned in the problem statement

Value of k: 12  
Training and test data are created by randomly splitting data in 66:34 ratio.  
Regressor accuracy is generally >**85**%

<hr>

## Naive Bayes Classifier
Mushroom dataset in .csv format.  
Downloaded from [Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)  
Mushroom Dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled
mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely
poisonous, or of unknown edibility and not recommended.  
The columns in this dataset are -  
1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
4. bruises?: bruises=t,no=f
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
6. gill-attachment: attached=a,descending=d,free=f,notched=n
7. gill-spacing: close=c,crowded=w,distant=d
8. gill-size: broad=b,narrow=n
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,
white=w,yellow=y
10. stalk-shape: enlarging=e,tapering=t
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
16. veil-type: partial=p,universal=u
17. veil-color: brown=n,orange=o,white=w,yellow=y
18. ring-number: none=n,one=o,two=t
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

The code is contained in [naive-bayes.py](naive-bayes-classifier/naive-bayes.py)  
It contains 6 functions -
1. loadDataset - Loads the dataset from a .csv file.
2. split data by classes - Split training data according to class labels
3. calculate probabilities - Calculating dependent probabilities for each feature given a particular class
label.
4. calculate z - Calculate the scaling factor Z
5. predict class - Returns the most likely class by calculating argmax for each class label
6. main - driver function

Instances with missing attributes are skipped.  
Training - top 4000 instances  
Test - Remaining 1644 instances  
Classifier accuracy is **84.97**%
