# EXNO2
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT:
```
from google.colab import drive
drive.mount('/content/drive')
```
![image](https://github.com/user-attachments/assets/6db5a664-1c93-4a29-9932-c099bbbde697)


```
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```


```
df = pd.read_csv("/content/drive/MyDrive/Data_Science/titanic_dataset.csv")
df
```
![image](https://github.com/user-attachments/assets/65a8eaf7-5c4c-466e-acc2-98824bfd60c8)


```
df.info()
```
![image](https://github.com/user-attachments/assets/09fab734-7662-4f29-b49d-e71e4f8063eb)


```
#DISPLAY NO OF ROWS AND COLUMNS
df.shape
```
![image](https://github.com/user-attachments/assets/9d92a346-b879-4eba-a8af-227520e4c7f7)


```
#SET PASSENGER ID AS INDEX COLUMN
df.set_index('PassengerId', inplace=True)
```


```
df.describe()
```
![image](https://github.com/user-attachments/assets/576e5bb4-da6e-4c97-a81a-38384e35663f)

## CATEGORICAL DATA ANALYSIS :
```
# USE VALUE COUNT FUNCTION AND PERFROM CATEGORICAL ANALYSIS
df. nunique()
```
![image](https://github.com/user-attachments/assets/c4f3582a-a521-49f7-b62b-37873c9f63a9)

## Survival Count
```
df['Survived'].value_counts()
```
![image](https://github.com/user-attachments/assets/efa4966e-2a1e-4347-b4b1-793d6aec2e00)


```
per = (df['Survived'].value_counts() / df.shape[0] * 100).round(2)
```

## Passenger Class Distribution
```
df['Pclass'].value_counts()
```
![image](https://github.com/user-attachments/assets/7836b3f7-2f20-424e-9eb5-e42333112d0d)

## Gender Distribution
```
df['Sex'].value_counts()
```
![image](https://github.com/user-attachments/assets/28e61c3a-c12d-4579-84e6-b910ee0db1cf)

## UNIVARIATE ANALYSIS :
```
# USE COUNTPLOT AND PERFORM UNIVARIATE ANALYSIS FOR THE "SURVIVED" COLUMN IN TITANIC DATASET
sns.countplot(data=df, x='Survived')
```
![image](https://github.com/user-attachments/assets/1af9dd83-7273-4afe-a6c2-a6ebbf978217)


```
# IDENTIFY UNIQUE VALUES IN "PASSENGER CLASS" COLUMN
df['Pclass'].unique()
```
![image](https://github.com/user-attachments/assets/049ebd91-e535-4179-9eec-f5d1477b63bf)


```
# RENAMING COLUMN
df.rename(columns = {'Sex':'Gender'}, inplace = True)
df
```
![image](https://github.com/user-attachments/assets/c70016ec-86c3-4930-892f-67a8189a5b03)

## BIVARIATE ANALYSIS :
```
# USE CATPLOT METHOD FOR BIVARIATE ANALYSIS
sns.catplot(data=df, x='Gender', kind='count', height=5, aspect=0.7)
plt.title('Count of Passengers by Gender')
plt.show()
```
![image](https://github.com/user-attachments/assets/899c90e4-0dc5-45fb-aeab-1ab562c93efa)


```
sns.catplot(data=df, x='Pclass', y='Survived', kind='point', height=5, aspect=0.7)
plt.title('Survival Rate by Passenger Class')
plt.show()
```
![image](https://github.com/user-attachments/assets/0ced13c9-48f7-4900-89e1-dedeaf9530b0)


```
sns.catplot(data=df, x='Pclass', y='Age', kind='violin', height=5, aspect=0.7)
plt.title('Age Distribution by Passenger Class')
plt.show()
```
![image](https://github.com/user-attachments/assets/c513e6e6-65fb-4a32-909d-2ad28828da31)


```
sns.catplot(data=df, x='Survived', y='Age', kind='strip', height=5, aspect=0.7)
plt.title('Age vs Survival Status')
plt.show()
```
![image](https://github.com/user-attachments/assets/dd273c76-cde0-4807-ae13-bce6b2e834a3)


```
sns.catplot(data=df, x='Pclass', y='Age', kind='swarm', height=5, aspect=0.7)
plt.title('Age vs Passenger Class')
plt.show()
```
![image](https://github.com/user-attachments/assets/06bd331f-fb83-4382-b82c-11df285094a1)


```
sns.scatterplot(data=df, x='Age', y='Fare')
plt.title('Age vs Fare')
plt.show()
```
![image](https://github.com/user-attachments/assets/c24c268e-5cb4-4ea4-be20-d429e4e9da56)


```
sns.lmplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title('Age vs Fare with Regression Line')
plt.show()
```
![image](https://github.com/user-attachments/assets/125426a6-289c-4337-846e-511bbea58604)


```
sns.jointplot(data=df, x='Age', y='Fare', hue='Survived')
plt.show()
```
![image](https://github.com/user-attachments/assets/5a842f4f-23a5-4432-a911-53a400cdeb33)


```
fig, ax1 = plt.subplots(figsize=(8,5))
graph = sns.countplot(data=df, x='Survived', hue='Gender')
graph.set_xticklabels(graph.get_xticklabels())
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2, height + 20.8,height ,ha="left")

plt.title('Countplot of Survival by Gender')
plt.show()
```
![image](https://github.com/user-attachments/assets/e5f5a3e8-c6b1-45f7-959d-269234aec920)


```
# USE BOXPLOT METHOD TO ANALYZE AGE AND SURVIVED COLUMN
sns.boxplot(data=df, x='Survived', y='Age')
plt.title('Age Distribution by Survival Status')
plt.show()
```
![image](https://github.com/user-attachments/assets/439a6020-2689-4062-87b0-169eb76493b4)

## MULTIVARIATE ANALYSIS :
```
# USE BOXPLOT METHOD AND ANALYZE THREE COLUMNS(PCLASS,AGE,GENDER)
sns.boxplot(data=df, x='Pclass', y='Age', hue='Gender')
plt.title('Age Distribution by Pclass and Gender')
plt.show()
```
![image](https://github.com/user-attachments/assets/2c01a333-2791-43e6-9ce1-333117c021b8)


```
# USE CATPLOT METHOD AND ANALYZE THREE COLUMNS(PCLASS,SURVIVED,GENDER)
sns.catplot(data=df, col='Survived', x='Gender', hue='Pclass', kind='count', height=5, aspect=1)
plt.title('Survival Rate by Pclass and Gender')
plt.show()
```
![image](https://github.com/user-attachments/assets/b5cf03da-9603-40c4-8424-90bd97bfd223)


```
# IMPLEMENT HEATMAP AND PAIRPLOT FOR THE DATASET
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title('Correlation Heatmap')
plt.show()
```
![image](https://github.com/user-attachments/assets/7bad146a-92d2-4b1f-b26c-a6b22ba1e287)


```
sns.pairplot(df)
```
![image](https://github.com/user-attachments/assets/c875815c-b1f1-4c98-87f0-b34ace593d11)


# RESULT:
      Thus, Exploratory Data Analysis on the given dataset is successfully performed.
