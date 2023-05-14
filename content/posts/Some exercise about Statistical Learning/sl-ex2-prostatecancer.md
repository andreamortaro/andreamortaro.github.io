+++
title = "SL2: Analysis of Prostate Cancer dataset - linear regression model"
author = "Andrea Mortaro"
layout = "notebook_toc"
showDate = false
weight = 5
draft = "false"
summary = " "
+++

{{< katex >}}


> <h4>Aim of the analysis</h4> We want to examinate correlation between the level of prostate-specific antigen and a number of clinical parameters in men, who are about to receive a prostatectomy.

```python
# data analysis and wrangling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


import statsmodels.api as sm

from IPython.display import Image # to visualize images
from tabulate import tabulate # to create tables

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/prostate-data/prostate.data
    /kaggle/input/prostate-data/tab.png
    /kaggle/input/prostate-data/tab2.png

<h4>1. Open the <a rel="nofollow" href="https://web.stanford.edu/~hastie/ElemStatLearn/" hreflang="eng" target="_blank">webpage of the book “The Elements of Statistical Learning”</a>, go to the “Data” section and download the info and data files for the dataset called Prostate</h4>

<h4>2. Open the file `prostate.info.txt`</h4>

* Hint: please, refer also to Section 3.2.1 (page 49) of the book “The Elements of Statistical Learning” to gather this information

* How many predictors are present in the dataset? What are those names?
    > There are 8 predictors:<br>
    > <t>       1. lcavol (log cancer volume)<br>
    >        2. lweight (log prostate weight)<br>
    >        3. age<br>
    >        4. lbph (log of the amount of benign prostatic hyperplasia)<br>
    >        5. svi (seminal veiscle invasion)<br>
    >        6. lcp (log of capsular penetration)<br>
    >        7. gleason (Gleason score)<br>
    >        8. pgg45 (percent of Gleason scores 4 or 5)
* How many responses are present in the dataset? What are their names?
    > There is one response:
            1. lpsa (log of prostate-specific antigen)
* How did the authors split the dataset in training and test set?
    > They randomly split the dataset (containing 97 observations, in `prostate.data`) into a training set of size 67 and a test set of size 30. <br> In the file `prostate.data` there is a column `train` which is of boolean type in order to distinguish if an observation is used (T) or not (F) to train the model.

<h4>3. Open the file `prostate.data` by a text editor or a spreadsheet and have a quick look at the data</h4>

* How many observations are present?
    > There are 97 observations in total.
* Which is the symbol used to separate the columns?
    > To separate the columns there is the escape character `\t` tab.

<h4>4. Open Kaggle, generate a new kernel and give it the name “SL_EX2_ProstateCancer_Surname”</h4>

> <h4>✅</h4>

<h4>5. Add the dataset `prostate.data` to the kernel</h4>

* Hint: See the Add Dataset button on the right
* Hint: use import option “Convert tabular files to csv”

> <h4>✅</h4>


<h4>6. Run the first cell of the kernel to check if the data file is present in folder ../input</h4>

> <h4>✅</h4>

<h4>7. Add to the first cell new lines to load the following libraries: seaborn, matplotlib.pyplot, sklearn.linear_model.LinearRegression</h4>

{{< alert >}}
<b>Tip:</b> We import also `pandas`.
{{< /alert >}}

<h4>8. Add a Markdown cell on top of the notebook, copy and paste in it the text of this exercise and provide in the same cell the answers to the questions that you get step-by-step.</h4>

> <h4>✅</h4>

<h4>9. Load the Prostate Cancer dataset into a Pandas DataFrame variable called "data"</h4>

* How can you say Python to use the right separator between columns?
    > I need to specify `sep='\t` into `read_csv` method in order to load the dataset.

## **Data acquisition**


```python
# Load the Prostate Cancer dataset
data = pd.read_csv('../input/prostate-data/prostate.data',sep='\t')
# data.info() # to check if it is correct
```

<h4>10. Display the number of rows and columns of variable data</h4>


```python
[num_rows,num_columns]=data.shape;

print(f"The number of rows is {num_rows} and the number of columns is {num_columns}.")
```

    The number of rows is 97 and the number of columns is 11.


<h4>11. Show the first 5 rows of the dataset</h4>


```python
data.head(5)
```




<div style="width:100%;overflow:scroll;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style='text-align:center; vertical-align:middle'></th>
      <th style='text-align:center; vertical-align:middle'>Unnamed: 0</th>
      <th style='text-align:center; vertical-align:middle'>lcavol</th>
      <th style='text-align:center; vertical-align:middle'>lweight</th>
      <th style='text-align:center; vertical-align:middle'>age</th>
      <th style='text-align:center; vertical-align:middle'>lbph</th>
      <th style='text-align:center; vertical-align:middle'>svi</th>
      <th style='text-align:center; vertical-align:middle'>lcp</th>
      <th style='text-align:center; vertical-align:middle'>gleason</th>
      <th style='text-align:center; vertical-align:middle'>pgg45</th>
      <th style='text-align:center; vertical-align:middle'>lpsa</th>
      <th style='text-align:center; vertical-align:middle'>train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>-0.579818</td>
      <td style='text-align:center; vertical-align:middle'>2.769459</td>
      <td style='text-align:center; vertical-align:middle'>50</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>-0.430783</td>
      <td style='text-align:center; vertical-align:middle'>T</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>-0.994252</td>
      <td style='text-align:center; vertical-align:middle'>3.319626</td>
      <td style='text-align:center; vertical-align:middle'>58</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>-0.162519</td>
      <td style='text-align:center; vertical-align:middle'>T</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>-0.510826</td>
      <td style='text-align:center; vertical-align:middle'>2.691243</td>
      <td style='text-align:center; vertical-align:middle'>74</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>20</td>
      <td style='text-align:center; vertical-align:middle'>-0.162519</td>
      <td style='text-align:center; vertical-align:middle'>T</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>-1.203973</td>
      <td style='text-align:center; vertical-align:middle'>3.282789</td>
      <td style='text-align:center; vertical-align:middle'>58</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>-0.162519</td>
      <td style='text-align:center; vertical-align:middle'>T</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>0.751416</td>
      <td style='text-align:center; vertical-align:middle'>3.432373</td>
      <td style='text-align:center; vertical-align:middle'>62</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0.371564</td>
      <td style='text-align:center; vertical-align:middle'>T</td>
    </tr>
  </tbody>
</table>
</div>



<h4>12. Remove the first column of the dataset which contains observation indices</h4>


```python
print("* Before to drop the first column:")
data.info()
#data1 = data1.drop(columns='Unnamed: 0')
#data1 = data1.drop(labels=['Unnamed: 0'],axis=1)

print("\n* After having dropped the first column:")
data = data.drop(data.columns[0],axis=1) # without specifying the name of the variable (axis=0 indicates rows, axis=1 indicates columns)
data.info()

data['train'].value_counts()
```

    * Before to drop the first column:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 97 entries, 0 to 96
    Data columns (total 11 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Unnamed: 0  97 non-null     int64  
     1   lcavol      97 non-null     float64
     2   lweight     97 non-null     float64
     3   age         97 non-null     int64  
     4   lbph        97 non-null     float64
     5   svi         97 non-null     int64  
     6   lcp         97 non-null     float64
     7   gleason     97 non-null     int64  
     8   pgg45       97 non-null     int64  
     9   lpsa        97 non-null     float64
     10  train       97 non-null     object 
    dtypes: float64(5), int64(5), object(1)
    memory usage: 8.5+ KB
    
    
    * After having dropped the first column:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 97 entries, 0 to 96
    Data columns (total 10 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   lcavol   97 non-null     float64
     1   lweight  97 non-null     float64
     2   age      97 non-null     int64  
     3   lbph     97 non-null     float64
     4   svi      97 non-null     int64  
     5   lcp      97 non-null     float64
     6   gleason  97 non-null     int64  
     7   pgg45    97 non-null     int64  
     8   lpsa     97 non-null     float64
     9   train    97 non-null     object 
    dtypes: float64(5), int64(4), object(1)
    memory usage: 7.7+ KB


    T    67
    F    30
    Name: train, dtype: int64


{{< alert >}}
<b>Warning:</b> Keep attention to do not run the above cell twice. Otherwise it will drop again the first column.
{{< /alert >}}


<h4>13. Save column train in a new variable called "train" and having type `Series` (the Pandas data structure used to represent DataFrame columns), then drop the column train from the data DataFrame</h4>


```python
# Save "train" column in a Pandas Series variable
train = data['train']
# train = pd.Series(data['train'])

# Drop "train" variable from data
data = data.drop(columns=['train'])
```
{{< alert >}}
<b>Warning:</b> Keep attention to do not run the above cell twice. In this case you have already dropped 'train'.
{{< /alert >}}

<h4>14. Save column lpsa in a new variable called "lpsa" and having type `Series` (the Pandas data structure used to represent DataFrame columns), then drop the column lpsa from the data DataFrame and save the result in a new DataFrame called predictors</h4>

* How many predictors are available?
    > There are 8 predictors available for each one of the 97 observations.


```python
# Save "lpsa" column in a Pandas Series variable
lpsa = data['lpsa']
# lpsa = pd.Series(data['lpsa'])

# Drop "train" variable from data
data = data.drop(columns=['lpsa'])
predictors = data
predictors.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 97 entries, 0 to 96
    Data columns (total 8 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   lcavol   97 non-null     float64
     1   lweight  97 non-null     float64
     2   age      97 non-null     int64  
     3   lbph     97 non-null     float64
     4   svi      97 non-null     int64  
     5   lcp      97 non-null     float64
     6   gleason  97 non-null     int64  
     7   pgg45    97 non-null     int64  
    dtypes: float64(4), int64(4)
    memory usage: 6.2 KB


<h4>15. Check the presence of missing values in the `data` variable</h4>
Since all the columns are numerical variables, we have not to distinguish variables between numerical and categorical kind.

* How many missing values are there? In which columns?
    > We have no missing values.
* Which types do the variable have?
    > `lcavol`,`lweight`,`lbph` and`lcp` are float64, while `age`,`svi`,`gleason` and `pgg45` are int64


```python
print("For each variable in data, we have no missing values:\n\n",data.isna().sum())
```

    For each variable in data, we have no missing values:
    
    lcavol     0
    lweight    0
    age        0
    lbph       0
    svi        0
    lcp        0
    gleason    0
    pgg45      0
    dtype: int64


<h4>16. Show histograms of all variables in a single figure</h4>

* Use argument figsize to enlarge the figure if needed


```python
fig = plt.figure()

predictors.hist(grid=True, figsize=(20,8), layout = (2,4))

fig.tight_layout()
plt.suptitle("Not Standardized data", fontsize=25)

fig.show()
```


    <Figure size 432x288 with 0 Axes>



![png](/posts/sl-ex2-prostatecancer/output_28_1.png)


<h4>17. Show the basic statistics (min, max, mean, quartiles, etc. for each variable) in data</h4>


```python
data.describe()
```




<div style="width:100%;overflow:scroll;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style='text-align:center; vertical-align:middle'></th>
      <th style='text-align:center; vertical-align:middle'>lcavol</th>
      <th style='text-align:center; vertical-align:middle'>lweight</th>
      <th style='text-align:center; vertical-align:middle'>age</th>
      <th style='text-align:center; vertical-align:middle'>lbph</th>
      <th style='text-align:center; vertical-align:middle'>svi</th>
      <th style='text-align:center; vertical-align:middle'>lcp</th>
      <th style='text-align:center; vertical-align:middle'>gleason</th>
      <th style='text-align:center; vertical-align:middle'>pgg45</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>count</th>
      <td style='text-align:center; vertical-align:middle'>97.000000</td>
      <td style='text-align:center; vertical-align:middle'>97.000000</td>
      <td style='text-align:center; vertical-align:middle'>97.000000</td>
      <td style='text-align:center; vertical-align:middle'>97.000000</td>
      <td style='text-align:center; vertical-align:middle'>97.000000</td>
      <td style='text-align:center; vertical-align:middle'>97.000000</td>
      <td style='text-align:center; vertical-align:middle'>97.000000</td>
      <td style='text-align:center; vertical-align:middle'>97.000000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>mean</th>
      <td style='text-align:center; vertical-align:middle'>1.350010</td>
      <td style='text-align:center; vertical-align:middle'>3.628943</td>
      <td style='text-align:center; vertical-align:middle'>63.865979</td>
      <td style='text-align:center; vertical-align:middle'>0.100356</td>
      <td style='text-align:center; vertical-align:middle'>0.216495</td>
      <td style='text-align:center; vertical-align:middle'>-0.179366</td>
      <td style='text-align:center; vertical-align:middle'>6.752577</td>
      <td style='text-align:center; vertical-align:middle'>24.381443</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>std</th>
      <td style='text-align:center; vertical-align:middle'>1.178625</td>
      <td style='text-align:center; vertical-align:middle'>0.428411</td>
      <td style='text-align:center; vertical-align:middle'>7.445117</td>
      <td style='text-align:center; vertical-align:middle'>1.450807</td>
      <td style='text-align:center; vertical-align:middle'>0.413995</td>
      <td style='text-align:center; vertical-align:middle'>1.398250</td>
      <td style='text-align:center; vertical-align:middle'>0.722134</td>
      <td style='text-align:center; vertical-align:middle'>28.204035</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>min</th>
      <td style='text-align:center; vertical-align:middle'>-1.347074</td>
      <td style='text-align:center; vertical-align:middle'>2.374906</td>
      <td style='text-align:center; vertical-align:middle'>41.000000</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>6.000000</td>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>25%</th>
      <td style='text-align:center; vertical-align:middle'>0.512824</td>
      <td style='text-align:center; vertical-align:middle'>3.375880</td>
      <td style='text-align:center; vertical-align:middle'>60.000000</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>-1.386294</td>
      <td style='text-align:center; vertical-align:middle'>6.000000</td>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>50%</th>
      <td style='text-align:center; vertical-align:middle'>1.446919</td>
      <td style='text-align:center; vertical-align:middle'>3.623007</td>
      <td style='text-align:center; vertical-align:middle'>65.000000</td>
      <td style='text-align:center; vertical-align:middle'>0.300105</td>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>-0.798508</td>
      <td style='text-align:center; vertical-align:middle'>7.000000</td>
      <td style='text-align:center; vertical-align:middle'>15.000000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>75%</th>
      <td style='text-align:center; vertical-align:middle'>2.127041</td>
      <td style='text-align:center; vertical-align:middle'>3.876396</td>
      <td style='text-align:center; vertical-align:middle'>68.000000</td>
      <td style='text-align:center; vertical-align:middle'>1.558145</td>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>1.178655</td>
      <td style='text-align:center; vertical-align:middle'>7.000000</td>
      <td style='text-align:center; vertical-align:middle'>40.000000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>max</th>
      <td style='text-align:center; vertical-align:middle'>3.821004</td>
      <td style='text-align:center; vertical-align:middle'>4.780383</td>
      <td style='text-align:center; vertical-align:middle'>79.000000</td>
      <td style='text-align:center; vertical-align:middle'>2.326302</td>
      <td style='text-align:center; vertical-align:middle'>1.000000</td>
      <td style='text-align:center; vertical-align:middle'>2.904165</td>
      <td style='text-align:center; vertical-align:middle'>9.000000</td>
      <td style='text-align:center; vertical-align:middle'>100.000000</td>
    </tr>
  </tbody>
</table>
</div>



<h4>18. Generate a new DataFrame called dataTrain and containing only the rows of data in which the train variable has value “T”</h4>

* Hint: use the loc attribute of DataFrame to access a groups of rows and columns by label(s) or boolean arrays
* How many rows and columns does dataTrain have?


```python
dataTrain = data.loc[train == 'T'] # Obviously, len(idx)==len(dataTrain) is True!

# # Alternative way:
# # 1. Get the indexes corresponding to train ==  'T'
# idxTrain = train.loc[train == 'T'].index.tolist() 
# # 2. Access to interesting rows with .iloc()
# dataTrain = data.iloc[idxTrain]

print(f"dataTrain contains {dataTrain.shape[0]} rows and {dataTrain.shape[1]} columns.")
```

    dataTrain contains 67 rows and 8 columns.


<h4>19. Generate a new DataFrame called dataTest and containing only the rows of data in which the train variable has value “F”</h4>

* How many rows and columns does dataTest have?


```python
dataTest = data.loc[train == 'F']
```

<h4>20. Generate a new Series called lpsaTrain and containing only the values of variable lpsa in which the train variable has value “T”</h4>

* How many valuses does lpsaTrain have?


```python
# Create a new Series variable
lpsaTrain = lpsa.loc[train == 'T']

# # Another way to define it
# data_all = pd.read_csv('../input/prostate-data/prostate.data',sep='\t')
# lpsaTrain = data_all.loc[train == 'T']['lpsa']

# # To check if it is correct:
# idxTrain = train.loc[train == 'T'].index.tolist()
# lpsaTrain == lpsa.iloc[idxTrain]

print(f"lpsaTrain has {lpsaTrain.shape[0]} values.")
```

    lpsaTrain has 67 values.


<h4>21. Generate a new Series called lpsaTest and containing only the values of variable lpsa in which the train variable has value “F”</h4>

* How many valuses does lpsaTest have?


```python
lpsaTest = lpsa.loc[train == 'F']

# # To check if it is correct: 
# len(lpsaTest) == len(data)-len(lpsaTrain)

print(f"lpsaTrain has {lpsaTest.shape[0]} values.")
```

    lpsaTrain has 30 values.


<h4>22. Show the correlation matrix among all the variables in dataTrain</h4>

* Hint: use the correct method in DataFrame
* Hint: check if the values in the matrix correspond to those in Table 3.1 of the book


```python
# Create correlation matrix
corrM = dataTrain.corr().round(decimals = 3) # As in the book, I plot values up to 3 decimals

# Display only the lower diagonal correlation matrix
lowerM = np.tril(np.ones(corrM.shape), k=-1) # Lower matrix of ones. (for k=0 I include also the main diagonal)
cond = lowerM.astype(bool) # Create a matrix of false, except in lowerM
corrM = corrM.where(cond, other='') # .where() replaces values with other where the condition is False.

corrM
```




<div style="width:100%;overflow:scroll;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style='text-align:center; vertical-align:middle'></th>
      <th style='text-align:center; vertical-align:middle'>lcavol</th>
      <th style='text-align:center; vertical-align:middle'>lweight</th>
      <th style='text-align:center; vertical-align:middle'>age</th>
      <th style='text-align:center; vertical-align:middle'>lbph</th>
      <th style='text-align:center; vertical-align:middle'>svi</th>
      <th style='text-align:center; vertical-align:middle'>lcp</th>
      <th style='text-align:center; vertical-align:middle'>gleason</th>
      <th style='text-align:center; vertical-align:middle'>pgg45</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>lcavol</th>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>lweight</th>
      <td style='text-align:center; vertical-align:middle'>0.3</td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>age</th>
      <td style='text-align:center; vertical-align:middle'>0.286</td>
      <td style='text-align:center; vertical-align:middle'>0.317</td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>lbph</th>
      <td style='text-align:center; vertical-align:middle'>0.063</td>
      <td style='text-align:center; vertical-align:middle'>0.437</td>
      <td style='text-align:center; vertical-align:middle'>0.287</td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>svi</th>
      <td style='text-align:center; vertical-align:middle'>0.593</td>
      <td style='text-align:center; vertical-align:middle'>0.181</td>
      <td style='text-align:center; vertical-align:middle'>0.129</td>
      <td style='text-align:center; vertical-align:middle'>-0.139</td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>lcp</th>
      <td style='text-align:center; vertical-align:middle'>0.692</td>
      <td style='text-align:center; vertical-align:middle'>0.157</td>
      <td style='text-align:center; vertical-align:middle'>0.173</td>
      <td style='text-align:center; vertical-align:middle'>-0.089</td>
      <td style='text-align:center; vertical-align:middle'>0.671</td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>gleason</th>
      <td style='text-align:center; vertical-align:middle'>0.426</td>
      <td style='text-align:center; vertical-align:middle'>0.024</td>
      <td style='text-align:center; vertical-align:middle'>0.366</td>
      <td style='text-align:center; vertical-align:middle'>0.033</td>
      <td style='text-align:center; vertical-align:middle'>0.307</td>
      <td style='text-align:center; vertical-align:middle'>0.476</td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'></td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>pgg45</th>
      <td style='text-align:center; vertical-align:middle'>0.483</td>
      <td style='text-align:center; vertical-align:middle'>0.074</td>
      <td style='text-align:center; vertical-align:middle'>0.276</td>
      <td style='text-align:center; vertical-align:middle'>-0.03</td>
      <td style='text-align:center; vertical-align:middle'>0.481</td>
      <td style='text-align:center; vertical-align:middle'>0.663</td>
      <td style='text-align:center; vertical-align:middle'>0.757</td>
      <td style='text-align:center; vertical-align:middle'></td>
    </tr>
  </tbody>
</table>
</div>




```python
# We can compare the above correlation matrix with the one in the book:
Image("../input/prostate-data/tab.png")
```




![png](/posts/sl-ex2-prostatecancer/output_41_0.png)



<h4>23. Drop the column lpsa from the `dataTrain` DataFrame and save the result in a new DataFrame called `predictorsTrain`</h4>

{{< alert >}}
<b>Warning:</b> I can not drop `lpsa` from `dataTrain`, because I have already done it!
{{< /alert >}}

In fact:
- at step 14. I dropped `lpsa` from `data`
- at step 18. I created `dataTrain` from `data`, by selecting certain rows. So at this step `dataTrain` does not contain `lpsa`.


```python
# predictorsTrain = dataTrain.drop['lpsa']

predictorsTrain = dataTrain
```

<h4>24. Drop the column `lpsa` from the dataTest DataFrame and save the result in a new DataFrame called `predictorsTest`</h4>


```python
dataTest.columns.tolist()

predictorsTest = dataTest
```
{{< alert >}}
<b>Warning:</b> I can not drop `lpsa` from `dataTest`, because I have already done it!
{{< /alert >}}


In fact:
- at step 14. I dropped `lpsa` from `data`
- at step 19. I created `dataTest` from `data`, by selecting certain rows. So at this step `dataTest` does not contain `lpsa`.

<h4>25. Generate a new DataFrame called `predictorsTrain_std` and containing the standardized variables of DataFrame `predictorsTrain`</h4>

* Hint: compute the mean of each column and save them in variable `predictorsTrainMeans`
* Hint: compute the standard deviation of each column and save them in variable `predictorsTrainStds`
* Hint: compute the standardization of each variable by the formula:
        ```
        \\[\\frac{predictorsTrain - predictorsTrainMeans}{predictorsTrainStd}\\]
        ```

```python
predictorsTrainMeans = predictorsTrain.mean()
predictorsTrainStds = predictorsTrain.std()
predictorsTrain_std = (predictorsTrain - predictorsTrainMeans)/predictorsTrainStds # standardized cariables of predictorTrain

predictorsTrain_std

# Standardizing makes it easier to compare scores, even if those scores were measured on different scales.
# It also makes it easier to read results from regression analysis and ensures that all variables contribute to a scale when added together.
```




<div style="width:100%;overflow:scroll;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style='text-align:center; vertical-align:middle'></th>
      <th style='text-align:center; vertical-align:middle'>lcavol</th>
      <th style='text-align:center; vertical-align:middle'>lweight</th>
      <th style='text-align:center; vertical-align:middle'>age</th>
      <th style='text-align:center; vertical-align:middle'>lbph</th>
      <th style='text-align:center; vertical-align:middle'>svi</th>
      <th style='text-align:center; vertical-align:middle'>lcp</th>
      <th style='text-align:center; vertical-align:middle'>gleason</th>
      <th style='text-align:center; vertical-align:middle'>pgg45</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>-1.523680</td>
      <td style='text-align:center; vertical-align:middle'>-1.797414</td>
      <td style='text-align:center; vertical-align:middle'>-1.965590</td>
      <td style='text-align:center; vertical-align:middle'>-0.995955</td>
      <td style='text-align:center; vertical-align:middle'>-0.533063</td>
      <td style='text-align:center; vertical-align:middle'>-0.836769</td>
      <td style='text-align:center; vertical-align:middle'>-1.031712</td>
      <td style='text-align:center; vertical-align:middle'>-0.896487</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>-1.857204</td>
      <td style='text-align:center; vertical-align:middle'>-0.643057</td>
      <td style='text-align:center; vertical-align:middle'>-0.899238</td>
      <td style='text-align:center; vertical-align:middle'>-0.995955</td>
      <td style='text-align:center; vertical-align:middle'>-0.533063</td>
      <td style='text-align:center; vertical-align:middle'>-0.836769</td>
      <td style='text-align:center; vertical-align:middle'>-1.031712</td>
      <td style='text-align:center; vertical-align:middle'>-0.896487</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>-1.468157</td>
      <td style='text-align:center; vertical-align:middle'>-1.961526</td>
      <td style='text-align:center; vertical-align:middle'>1.233468</td>
      <td style='text-align:center; vertical-align:middle'>-0.995955</td>
      <td style='text-align:center; vertical-align:middle'>-0.533063</td>
      <td style='text-align:center; vertical-align:middle'>-0.836769</td>
      <td style='text-align:center; vertical-align:middle'>0.378996</td>
      <td style='text-align:center; vertical-align:middle'>-0.213934</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>-2.025981</td>
      <td style='text-align:center; vertical-align:middle'>-0.720349</td>
      <td style='text-align:center; vertical-align:middle'>-0.899238</td>
      <td style='text-align:center; vertical-align:middle'>-0.995955</td>
      <td style='text-align:center; vertical-align:middle'>-0.533063</td>
      <td style='text-align:center; vertical-align:middle'>-0.836769</td>
      <td style='text-align:center; vertical-align:middle'>-1.031712</td>
      <td style='text-align:center; vertical-align:middle'>-0.896487</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>-0.452342</td>
      <td style='text-align:center; vertical-align:middle'>-0.406493</td>
      <td style='text-align:center; vertical-align:middle'>-0.366061</td>
      <td style='text-align:center; vertical-align:middle'>-0.995955</td>
      <td style='text-align:center; vertical-align:middle'>-0.533063</td>
      <td style='text-align:center; vertical-align:middle'>-0.836769</td>
      <td style='text-align:center; vertical-align:middle'>-1.031712</td>
      <td style='text-align:center; vertical-align:middle'>-0.896487</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>...</th>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>90</th>
      <td style='text-align:center; vertical-align:middle'>1.555621</td>
      <td style='text-align:center; vertical-align:middle'>0.998130</td>
      <td style='text-align:center; vertical-align:middle'>0.433703</td>
      <td style='text-align:center; vertical-align:middle'>-0.995955</td>
      <td style='text-align:center; vertical-align:middle'>-0.533063</td>
      <td style='text-align:center; vertical-align:middle'>-0.836769</td>
      <td style='text-align:center; vertical-align:middle'>-1.031712</td>
      <td style='text-align:center; vertical-align:middle'>-0.896487</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>91</th>
      <td style='text-align:center; vertical-align:middle'>0.981346</td>
      <td style='text-align:center; vertical-align:middle'>0.107969</td>
      <td style='text-align:center; vertical-align:middle'>-0.499355</td>
      <td style='text-align:center; vertical-align:middle'>0.872223</td>
      <td style='text-align:center; vertical-align:middle'>1.847952</td>
      <td style='text-align:center; vertical-align:middle'>-0.836769</td>
      <td style='text-align:center; vertical-align:middle'>0.378996</td>
      <td style='text-align:center; vertical-align:middle'>-0.384573</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>92</th>
      <td style='text-align:center; vertical-align:middle'>1.220657</td>
      <td style='text-align:center; vertical-align:middle'>0.525153</td>
      <td style='text-align:center; vertical-align:middle'>0.433703</td>
      <td style='text-align:center; vertical-align:middle'>-0.995955</td>
      <td style='text-align:center; vertical-align:middle'>1.847952</td>
      <td style='text-align:center; vertical-align:middle'>1.096538</td>
      <td style='text-align:center; vertical-align:middle'>0.378996</td>
      <td style='text-align:center; vertical-align:middle'>1.151171</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>93</th>
      <td style='text-align:center; vertical-align:middle'>2.017972</td>
      <td style='text-align:center; vertical-align:middle'>0.568193</td>
      <td style='text-align:center; vertical-align:middle'>-2.765355</td>
      <td style='text-align:center; vertical-align:middle'>-0.995955</td>
      <td style='text-align:center; vertical-align:middle'>1.847952</td>
      <td style='text-align:center; vertical-align:middle'>1.701433</td>
      <td style='text-align:center; vertical-align:middle'>0.378996</td>
      <td style='text-align:center; vertical-align:middle'>0.468618</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>95</th>
      <td style='text-align:center; vertical-align:middle'>1.262743</td>
      <td style='text-align:center; vertical-align:middle'>0.310118</td>
      <td style='text-align:center; vertical-align:middle'>0.433703</td>
      <td style='text-align:center; vertical-align:middle'>1.015748</td>
      <td style='text-align:center; vertical-align:middle'>1.847952</td>
      <td style='text-align:center; vertical-align:middle'>1.265298</td>
      <td style='text-align:center; vertical-align:middle'>0.378996</td>
      <td style='text-align:center; vertical-align:middle'>1.833724</td>
    </tr>
  </tbody>
</table>
<p>67 rows × 8 columns</p>
</div>



<h4>26. Show the histogram of each variables of predictorsTrain_std in a single figure</h4>

* Use argument figsize to enlarge the figure if needed
* Hint: which kind of difference can you see in the histograms?


```python
print("Now all the variables are centered at 0 and they variance equal to 1. So we can compare them in a better way.")
fig = plt.figure()

predictorsTrain_std.hist(grid=True, figsize=(20,8), layout = (2,4))

plt.suptitle("Standardized data", fontsize=25)
fig.tight_layout()

plt.show()
```

    Now all the variables are centered at 0 and they variance equal to 1. So we can compare them in a better way.



    <Figure size 432x288 with 0 Axes>



![png](/posts/sl-ex2-prostatecancer/output_51_2.png)


## **Linear Regression**

<h4>27. Generate a linear regression model using `predictorsTrain_std` as dependent variables and `lpsaTrain` as independent variable</h4>

* Hint: find a function for linear regression model learning in sklearn (**fit**)
* How do you set parameter **fit_intercept**? Why?
    > The parameter **`fit_intercept`** specifies whether to calculate the intercept for this model:
    > * If **False**, then the y-intercept to 0 (it is forced to 0);
    > * if **True**, then the y-intercept will be determined by the line of best fit (it's allowed to "fit" the y-axis).
    >
    > As default **`fit_intercept`** is **True** and it is good for us.

* How do you set parameter **normalize**? Why? Can this parameter be used to simplify the generation of the predictor matrix?
    > If **`fit_intercept` = False**, then the parameter **`normalize`** is ignored. If **`normalize` = True**, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.
    >
    > By default **`normalize` = False**.
    >
    > We have already standardized our variables, so we set it as **False**.


```python
# Create X_test
#predictorsTestMean = predictorsTest.mean()
predictorsTestStds = predictorsTest.std()
#predictorsTest_std = (predictorsTest - predictorsTestMeans)/predictorsTestStds # standardized cariables of predictorTrain
predictorsTest_std = (predictorsTest - predictorsTrainMeans)/predictorsTrainStds # standardized cariables of predictorTrain (BETTER WAY TO DO IT)
```


```python
# Prepare the independent and dependent variables for the model

# Independent variables
X_train = predictorsTrain_std
X_test = predictorsTest_std

# Dependent variable
y_train = lpsaTrain
y_test = lpsaTest
```


```python
linreg = LinearRegression() # we don't need to specify args, because the default ones are already good for us
linreg.fit(X_train,y_train)

# by default: fit_intercept = True and normalize = False
# This setting is good because we want to compute the intercept and we don't need to normalize X because we have already done it
```

LinearRegression()



## **Difference in setting up <kbd>fit_intercept</kbd> in LinearRegression()**


```python
# Difference in setting up fit_intercept

lr_fi_true = LinearRegression(fit_intercept=True)
lr_fi_false = LinearRegression(fit_intercept=False)

lr_fi_true.fit(X_train, y_train)
lr_fi_false.fit(X_train, y_train)

print('Intercept when fit_intercept=True : {:.5f}'.format(lr_fi_true.intercept_))
print('Intercept when fit_intercept=False : {:.5f}'.format(lr_fi_false.intercept_))

# FIGURE
# SOURCE: https://stackoverflow.com/questions/46779605/in-the-linearregression-method-in-sklearn-what-exactly-is-the-fit-intercept-par

# fig properties
row = 2
col = 4
width = 20
height = 8

# initialize the figure
fig, axes = plt.subplots(row, col,figsize=(width,height))

for ax,variable in zip(axes.flatten(),X_train.columns.tolist()):
    ax.scatter(X_train[variable],y_train, label='Actual points')
    
    ax.grid(color='grey', linestyle='-', linewidth=0.5)
    
    idx = X_train.columns.get_loc(variable) # get corresponding column index to access the right coeff
    
    lr_fi_true_yhat = np.dot(X_train[variable], lr_fi_true.coef_[idx]) + lr_fi_true.intercept_
    lr_fi_false_yhat = np.dot(X_train[variable], lr_fi_false.coef_[idx]) + lr_fi_false.intercept_
    
    ax.plot(X_train[variable], lr_fi_true_yhat, 'g--', label='fit_intercept=True')
    ax.plot(X_train[variable], lr_fi_false_yhat, 'r-', label='fit_intercept=False')

fig.tight_layout()

plt.show(fig) # force to show the plot after the print

```

    Intercept when fit_intercept=True : 2.45235
    Intercept when fit_intercept=False : 0.00000



![png](/posts/sl-ex2-prostatecancer/output_58_1.png)



```python
lr_fi_true = LinearRegression(fit_intercept=True)
lr_fi_true.fit(X_train, y_train)
print(lr_fi_true.coef_,"\n",lr_fi_true.intercept_,"\n\n")

lr_fi_false = LinearRegression(fit_intercept=False)
lr_fi_false.fit(X_train, y_train)
print(lr_fi_true.coef_,"\n",lr_fi_false.intercept_)
```

    [ 0.71640701  0.2926424  -0.14254963  0.2120076   0.30961953 -0.28900562
     -0.02091352  0.27734595] 
     2.4523450850746262 
    
    
    [ 0.71640701  0.2926424  -0.14254963  0.2120076   0.30961953 -0.28900562
     -0.02091352  0.27734595] 
     0.0


<h4>28. Show the parameters of the linear regression model computed above. Compare the parameters with those shown in Table 3.2 of the book (page 50)</h4>


```python
col = ['Term','Coefficient'] # headers

intercept_val = np.array([linreg.intercept_]).round(2)
coeff_val = linreg.coef_.round(2)
intercept_label = np.array(['Intercept'])
coeff_label = X_train.columns.tolist()

terms = np.concatenate((intercept_val,coeff_val), axis=0)     
coeffs = np.concatenate((intercept_label,coeff_label),axis=0)

table = np.column_stack((coeffs,terms))

print(tabulate(table, headers=col, tablefmt='fancy_grid'))
```

    ╒═══════════╤═══════════════╕
    │ Term      │   Coefficient │
    ╞═══════════╪═══════════════╡
    │ Intercept │          2.45 │
    ├───────────┼───────────────┤
    │ lcavol    │          0.72 │
    ├───────────┼───────────────┤
    │ lweight   │          0.29 │
    ├───────────┼───────────────┤
    │ age       │         -0.14 │
    ├───────────┼───────────────┤
    │ lbph      │          0.21 │
    ├───────────┼───────────────┤
    │ svi       │          0.31 │
    ├───────────┼───────────────┤
    │ lcp       │         -0.29 │
    ├───────────┼───────────────┤
    │ gleason   │         -0.02 │
    ├───────────┼───────────────┤
    │ pgg45     │          0.28 │
    ╘═══════════╧═══════════════╛



```python
# We can compare the above correlation matrix with the one in the book:
Image("../input/prostate-data/tab2.png")
```




![png](/posts/sl-ex2-prostatecancer/output_62_0.png)


<h4>29. Compute the coefficient of determination of the prediction</h4>

For coefficient of determination we mean \\(R^{2}\\).


```python
y_predicted = linreg.predict(X_test)
y_predicted
```




    array([1.96903844, 1.16995577, 1.26117929, 1.88375914, 2.54431886,
           1.93275402, 2.04233571, 1.83091625, 1.99115929, 1.32347076,
           2.93843111, 2.20314404, 2.166421  , 2.79456237, 2.67466879,
           2.18057291, 2.40211068, 3.02351576, 3.21122283, 1.38441459,
           3.41751878, 3.70741749, 2.54118337, 2.72969658, 2.64055575,
           3.48060024, 3.17136269, 3.2923494 , 3.11889686, 3.76383999])




```python
score = r2_score(y_test,y_predicted) # goodness of fit measure for linreg
score2 = mean_squared_error(y_test,y_predicted)

print(f"The coefficient of determination (i.e. R^2) of the prediction is {round(score,3)}\n\
The mean squared error is: {round(score2,3)}.\n\
The root of the mean squared error is {round(np.sqrt(score2),3)}")
```

    The coefficient of determination (i.e. R^2) of the prediction is 0.503
    The mean squared error is: 0.521.
    The root of the mean squared error is 0.722



```python
plt.figure(figsize=[7,5])
plt.scatter(X_test['lcavol'],y_test, marker='o', s = 50)
plt.scatter(X_test['lcavol'],y_predicted, marker='^', s = 50)
plt.title('Predicted and Real values of Y by using `lcavol`',fontsize=16)

plt.legend(labels = ['Test','Predicted'],loc = 'upper right')
plt.xlabel('lcavol',fontsize=12)
plt.ylabel('lpsa',fontsize=12)

plt.show()
```


![png](/posts/sl-ex2-prostatecancer/output_67_0.png)




```python
data_all = pd.read_csv('../input/prostate-data/prostate.data',sep='\t')
data_all = data_all.drop(labels = ['Unnamed: 0'],axis=1)
```


```python
featuresToPlot = data_all.columns.tolist()
dataToPlot = data_all[featuresToPlot]

pd.plotting.scatter_matrix(dataToPlot, alpha = 0.7, diagonal = 'kde', figsize=(10,10))

plt.show()
```


![png](/posts/sl-ex2-prostatecancer/output_70_0.png)


```python
plt.figure(figsize=[7,5])
plt.scatter(predictorsTrain_std['lweight'],lpsaTrain, marker='o', s = 50)
plt.title('Linear relationship btw `lweight` and  `lpsa`',fontsize=16)

plt.xlabel('lweight',fontsize=12)
plt.ylabel('lpsa',fontsize=12)

plt.show()

# in this case linear regression is nice!
```


![png](/posts/sl-ex2-prostatecancer/output_72_0.png)


We can see how these relationships are linear. We can see linearity from the plot.


```python
r2test = round(linreg.score(X_test,lpsaTest),2)
r2train = round(linreg.score(X_train,lpsaTrain),2)

print(f"Coefficient of determination for Test set is {r2test}\n\
Coefficient of determination for Test set is {r2train}") # it's higher because the model is created by using train set 
```

    Coefficient of determination for Test set is 0.5
    Coefficient of determination for Test set is 0.69


<h4>30. Compute the standard errors, the Z scores (Student’s t statistics) and the related p-values</h4>

* Hint: use library `statsmodels instead of sklearn
* Hint: compare the results with those in Table 3.2 of the book (page 50)


```python
y_predicted = linreg.predict(X_test)

X_trainC = sm.add_constant(X_train) # We need this in order to have an intercept
                        # otherwise we will no have the costant (it would be 0)

model = sm.OLS(y_train,X_trainC) # define the OLS (Ordinary Least Square - Linear Regression model)

results = model.fit() # fit the model

results.params # Coefficients of the model
results.summary(slim=True)

# Adjusted R^2 is the r^2 scaled by the number of parameters in the model
# F-statistics tells if the value of R^2 is significant or not.
```


<div style="width:100%;overflow:scroll;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<caption>OLS Regression Results</caption>
<tbody>
<tr>
  <th style='text-align:center; vertical-align:middle'>Dep. Variable:</th>
  <td style='text-align:center; vertical-align:middle'>lpsa</td>
  <th style='text-align:center; vertical-align:middle'>R-squared:</th>
  <td style='text-align:center; vertical-align:middle'> 0.694</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>Model:</th>
  <td style='text-align:center; vertical-align:middle'>OLS</td>
  <th style='text-align:center; vertical-align:middle'>Adj. R-squared:</th>
  <td style='text-align:center; vertical-align:middle'> 0.652</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>No. Observations:</th>
  <td style='text-align:center; vertical-align:middle'>67</td>
  <th style='text-align:center; vertical-align:middle'>F-statistic:</th>
  <td style='text-align:center; vertical-align:middle'> 16.47</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>Covariance Type:</th>
  <td style='text-align:center; vertical-align:middle'>nonrobust</td>
  <th style='text-align:center; vertical-align:middle'>Prob (F-statistic):</th>
  <td style='text-align:center; vertical-align:middle'>2.04e-12</td>
    </tr>
  </tbody>
</table>
</div>

<div style="width:100%;overflow:scroll;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
<thead>
<tr>
<td style='text-align:center; vertical-align:middle'></td>
  <th style='text-align:center; vertical-align:middle'>coef</th>
  <th style='text-align:center; vertical-align:middle'>std err</th>
  <th style='text-align:center; vertical-align:middle'>t</th>
  <th style='text-align:center; vertical-align:middle'>P>|t|</th>
  <th style='text-align:center; vertical-align:middle'>[0.025</th>
  <th style='text-align:center; vertical-align:middle'>0.975]</th>
</tr>
</thead>
<tbody>
<tr>
  <th style='text-align:center; vertical-align:middle'>const</th>
  <td style='text-align:center; vertical-align:middle'>2.4523</td>
  <td style='text-align:center; vertical-align:middle'>0.087</td>
  <td style='text-align:center; vertical-align:middle'>28.182</td>
  <td style='text-align:center; vertical-align:middle'>0.000</td>
  <td style='text-align:center; vertical-align:middle'>2.278</td>
  <td style='text-align:center; vertical-align:middle'>2.627</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>lcavol</th>
  <td style='text-align:center; vertical-align:middle'>0.7164</td>
  <td style='text-align:center; vertical-align:middle'>0.134</td>
  <td style='text-align:center; vertical-align:middle'>5.366</td>
  <td style='text-align:center; vertical-align:middle'>0.000</td>
  <td style='text-align:center; vertical-align:middle'>0.449</td>
  <td style='text-align:center; vertical-align:middle'>0.984</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>lweight</th>
  <td style='text-align:center; vertical-align:middle'>0.2926</td>
  <td style='text-align:center; vertical-align:middle'>0.106</td>
  <td style='text-align:center; vertical-align:middle'>2.751</td>
  <td style='text-align:center; vertical-align:middle'>0.008</td>
  <td style='text-align:center; vertical-align:middle'>0.080</td>
  <td style='text-align:center; vertical-align:middle'>0.506</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>age</th>
  <td style='text-align:center; vertical-align:middle'> -0.1425</td>
  <td style='text-align:center; vertical-align:middle'>0.102</td>
  <td style='text-align:center; vertical-align:middle'> -1.396</td>
  <td style='text-align:center; vertical-align:middle'>0.168</td>
  <td style='text-align:center; vertical-align:middle'> -0.347</td>
  <td style='text-align:center; vertical-align:middle'>0.062</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>lbph</th>
  <td style='text-align:center; vertical-align:middle'>0.2120</td>
  <td style='text-align:center; vertical-align:middle'>0.103</td>
  <td style='text-align:center; vertical-align:middle'>2.056</td>
  <td style='text-align:center; vertical-align:middle'>0.044</td>
  <td style='text-align:center; vertical-align:middle'>0.006</td>
  <td style='text-align:center; vertical-align:middle'>0.418</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>svi</th>
  <td style='text-align:center; vertical-align:middle'>0.3096</td>
  <td style='text-align:center; vertical-align:middle'>0.125</td>
  <td style='text-align:center; vertical-align:middle'>2.469</td>
  <td style='text-align:center; vertical-align:middle'>0.017</td>
  <td style='text-align:center; vertical-align:middle'>0.059</td>
  <td style='text-align:center; vertical-align:middle'>0.561</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>lcp</th>
  <td style='text-align:center; vertical-align:middle'> -0.2890</td>
  <td style='text-align:center; vertical-align:middle'>0.155</td>
  <td style='text-align:center; vertical-align:middle'> -1.867</td>
  <td style='text-align:center; vertical-align:middle'>0.067</td>
  <td style='text-align:center; vertical-align:middle'> -0.599</td>
  <td style='text-align:center; vertical-align:middle'>0.021</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>gleason</th>
  <td style='text-align:center; vertical-align:middle'> -0.0209</td>
  <td style='text-align:center; vertical-align:middle'>0.143</td>
  <td style='text-align:center; vertical-align:middle'> -0.147</td>
  <td style='text-align:center; vertical-align:middle'>0.884</td>
  <td style='text-align:center; vertical-align:middle'> -0.306</td>
  <td style='text-align:center; vertical-align:middle'>0.264</td>
</tr>
<tr>
  <th style='text-align:center; vertical-align:middle'>pgg45</th>
  <td style='text-align:center; vertical-align:middle'>0.2773</td>
  <td style='text-align:center; vertical-align:middle'>0.160</td>
  <td style='text-align:center; vertical-align:middle'>1.738</td>
  <td style='text-align:center; vertical-align:middle'>0.088</td>
  <td style='text-align:center; vertical-align:middle'> -0.042</td>
  <td style='text-align:center; vertical-align:middle'>0.597</td>
    </tr>
  </tbody>
</table>
</div>

<br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


```python
# We can compare the above correlation matrix with the one in the book:
Image("../input/prostate-data/tab2.png")
```


![png](/posts/sl-ex2-prostatecancer/output_78_0.png)


