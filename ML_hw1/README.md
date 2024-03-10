2021 Introduction to Machine Learning Program Assignment #1 - Naïve Bayes
TA: Evan Chang toosyou.tw@gmail.com
This programming assignment aims to help you understand the algorithm behind Naïve Bayes classifier and the basic workflow of machine learning.

Before we start
Join the discord server for TA support

Ask questions on it, and we shall reply.
Try not to ask for obvious answers or bug fixes.
Memes and chit-chat welcome.
Objective
There are two datasets that need to be analyzed. For each dataset, you have to do the following:

Data Input - 5%
Data Visualization - 15%
For mushroom dataset
Show the data distribution by value frequency of every feature.
For Iris dataset
Show the data distribution by average, standard deviation, and value frequency(binning might be needed) of every feature.
Split data based on their labels (targets) and show the data distribution of each feature again.
Data Preprocessing - 5% + (10%)
Drop features with any missing value.
Transform data format and shape so your model can process them.
Shuffle the data.
Bonus: any other transformation boosts the final performance. - (10%)
Model Construction - 20%
You must construct two Naïve Bayes classifiers for the two datasets.
You may use any package avaliable as long as the classifiers fit the following description.
Naïve Bayes divider 𝑀 in log-space:
𝑀(𝐪)=argmax𝑌∈𝕋[log𝑃(𝑌)+∑𝑚𝑖=1log𝑃(𝑋𝑖|𝑌)]
where 𝐪={𝑋1,𝑋2,...,𝑋𝑚} is a sample to be predicted, whose features are 𝑋1 to 𝑋𝑚. 𝕋 is the set of all possible labels.
For the mushroom dataset, whose features are all categorical, 𝑃(𝑋𝑖|𝑌) must be computed with and without Laplace smoothing for result comparison. - 10%
Without Laplace smoothing
𝑃(𝑋𝑖|𝑌)=𝑁(𝑋𝑖|𝑌)𝑁(𝑌)
Laplace smoothing
𝑃(𝑋𝑖|𝑌)=𝑁(𝑋𝑖|𝑌)+𝑘𝑁(𝑌)+𝑘𝜏
where 𝜏 is the number of all possible events of feature 𝑋𝑖
For Iris dataset, whose features are all numerical, assume 𝑃(𝑋𝑖|𝑌) follows a 1D-Normal(Gaussian) distribution. - 10%
𝑃(𝑋𝑖|𝑌)=1𝜎2𝜋√𝑒−(𝑥−𝜇)22𝜎2
where 𝜇,𝜎 are the mean and standard deviation of feature 𝑋𝑖 respectively, while label 𝑌 is determined.
Train-Test-Split - 10%
Two validation methods need to be implemented.
Holdout validation with the ratio 7:3
K-fold cross-validation with 𝐾=3
Obtain the final performance by averaging all folds’ performance.
Results - 20%
Obtain the performances of all experiment settings in tables by the following metrics:
Confusion matrix
Accuracy
Sensitivity(Recall)
Precision
Comparison & Conclusion - 5%
Questions - 25%
For the mushroom dataset
Show 𝑃(𝑋𝑠𝑡𝑎𝑙𝑘−𝑐𝑜𝑙𝑜𝑟−𝑏𝑒𝑙𝑜𝑤−𝑟𝑖𝑛𝑔|𝑌=𝑒) with and without Laplace smoothing by bar charts - 10%
For Iris dataset
What are the values of 𝜇 and 𝜎 of assumed 𝑃(𝑋𝑝𝑒𝑡𝑎𝑙_𝑙𝑒𝑛𝑔𝑡ℎ|𝑌=Iris Versicolour)? - 5%
Use a graph to show the probability density function of assumed 𝑃(𝑋𝑝𝑒𝑡𝑎𝑙_𝑙𝑒𝑛𝑔𝑡ℎ|𝑌=Iris Versicolour) - 10%
Data
1. Mushroom dataset
Data can be downloaded here:
https://archive.ics.uci.edu/ml/datasets/mushroom
Please NOTE that the first column is the label (edible=e, poisonous=p)
Data Set Information
This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ‘‘leaflets three, let it be’’ for Poisonous Oak and Ivy.
Attribute Information
cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
bruises?: bruises=t,no=f
odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
gill-attachment: attached=a,descending=d,free=f,notched=n
gill-spacing: close=c,crowded=w,distant=d
gill-size: broad=b,narrow=n
gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
stalk-shape: enlarging=e,tapering=t
stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r, missing=?
stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
veil-type: partial=p,universal=u
veil-color: brown=n,orange=o,white=w,yellow=y
ring-number: none=n,one=o,two=t
ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
2. Iris dataset
Data can be downloaded here:
https://archive.ics.uci.edu/ml/datasets/iris
Data Set Information
This is perhaps the best known database to be found in the pattern recognition literature. Fisher’s paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. Predicted attribute: class of iris plant. This is an exceedingly simple domain.
Attribute Information
sepal length in cm
sepal width in cm
petal length in cm
petal width in cm
class:
Iris Setosa
Iris Versicolour
Iris Virginica
Submission & Scoring Policy
Please submit a zip file, which contains the following, to the newE3 system.
Report
Explanation of how your code works.
All the content mentioned above.
Your name and student ID at the very beginning - 10%
Accept formats: HTML
From markdowns or jupyter notebooks.
Source codes
Accept languages: python3
Accept formats: .ipynb
Package-provided models are allowed
Your score will be determined mainly by the submitted report.
If there’s any problem with your code, TA might ask you (through email) to demo it. Otherwise, no demo is needed.
Scores will be adjusted at the end of the semester for them to fit the school regulations.
Plagiarizing is not allowed.
Plagiarizing is checked by MOSS and manually afterward.
You will get ZERO on that homework if you get caught the first time.
The second time, you’ll FAIL this class.
抄襲第一次作業零分，第二次當掉
Acknowledgments
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
Tools that might be useful
Jupyter Lab - Better data science experience
numpy - Math thingy
matplotlib - Plot thingy
pandas - Data thingy
scikit-learn - Machine Learning and stuff
