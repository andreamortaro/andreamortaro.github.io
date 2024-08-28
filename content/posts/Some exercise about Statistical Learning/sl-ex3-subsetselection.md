+++
title = "SL3: Analysis of Prostate Cancer dataset – variable subset selection"
author = "Andrea Mortaro"
layout = "single"
showDate = false
weight = 4
draft = "false"
summary = " "
+++

{{< katex >}}

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
import itertools # itertools functions create iterators for efficient looping
                 # itertools.combinations(p,r) creates r-length tuples, in sorted order, no repeated elements
                 # ex. combinations('ABCD', 2) = [AB AC AD BC BD CD]
        

#from IPython.display import Image # to visualize images
#from tabulate import tabulate # to create tables

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/prostate-data/tab4.png
    /kaggle/input/prostate-data/tab4-no-cap.png
    /kaggle/input/prostate-data/tab3.png
    /kaggle/input/prostate-data/prostate.data
    /kaggle/input/prostate-data/tab.png
    /kaggle/input/prostate-data/tab2.png


<h4>1. Open your kernel SL_EX2_ProstateCancer_Surname in Kaggle</h4>

<h4>2. Generate a copy called SL_EX3_SubsetSelection_Surname by the Fork button</h4>

## **Data acquisition**


```python
# Load the Prostate Cancer dataset
data = pd.read_csv('../input/prostate-data/prostate.data',sep='\t')
```


```python
# Save "train" and lpsa" columns into Pandas Series variables
train = data['train']
lpsa = data['lpsa']

# Drop "train" and lpsa" variable from data
data = data.drop(columns=['Unnamed: 0','lpsa','train'],axis=1)
```

## **Data pre-processing**


```python
## X VARIABLE:
# Split the data in train and test sets
dataTrain = data.loc[train == 'T'] # Obviously, len(idx)==len(dataTrain) is True!
dataTest = data.loc[train == 'F']

# Rename these two variables as "predictorsTrain" and "predictorsTest"
predictorsTrain = dataTrain
predictorsTest = dataTest


## Y VARIABLE:
# Split the "lpsa" in train and test sets
lpsaTrain = lpsa.loc[train == 'T']
lpsaTest = lpsa.loc[train == 'F']
```

<h3>Standardization</h3> 

\\[\\dfrac{predictors - predictorsMeans}{predictorsStd}\\]


```python
# Standardize "predictorsTrain"
predictorsTrainMeans = predictorsTrain.mean()
predictorsTrainStds = predictorsTrain.std()
predictorsTrain_std = (predictorsTrain - predictorsTrainMeans)/predictorsTrainStds # standardized variables of predictorTrain

# Standardize "predictorsTest" (using the mean and std of predictorsTrain, it's better!)
predictorsTest_std = (predictorsTest - predictorsTrainMeans)/predictorsTrainStds # standardized variables of predictorTest
```


<h3>Split into Training and Test sets</h3>


```python
## TRAINING SET
X_train = predictorsTrain_std
Y_train = lpsaTrain

## TEST SET
X_test = predictorsTest_std
Y_test = lpsaTest
```

## **Useful functions for the analysis of the Prostate_Cancer_dataset**

<h3>For Linear Regression</h3>

```python
# Useful functions in order to compute RSS, R_squared and Zscore.
def LinReg(X_train,Y_train,X_test,Y_test):
    
    # Create the linear model, fitting also the intecept (non-zero)
    model = LinearRegression(fit_intercept = True)
    
    # Train the model on training set
    model.fit(X_train,Y_train)
    
    # Stats on Training set
    Y_train_pred = model.predict(X_train)
    RSS_train = mean_squared_error(Y_train,Y_train_pred) * len(Y_train)
    R2_train = model.score(X_train,Y_train)
    
    # Stats on Test set
    Y_test_pred = model.predict(X_test)
    RSS_test = mean_squared_error(Y_test,Y_test_pred) * len(Y_test)
    R2_test = model.score(X_test,Y_test)
    
    return RSS_train, RSS_test, R2_train

def Zscore(X_train,Y_train):
    
    # fitting the model
    model = sm.OLS(Y_train, sm.add_constant(X_train)).fit()
    
    Zscores = model.tvalues[1:] # we don't want const
    min_Zscore = min(abs(Zscores))
    idx_min_Zscore = abs(Zscores).idxmin() # it's the nearest to zero, so the variable less significant!

    return Zscores, min_Zscore, idx_min_Zscore
```

<h3>To print some info</h3>


```python
## Print information about an iteration forward subset selection with subset size ncomb
# ncomb = number of the iteration
# features = remaining_features, selected_features (used to do the iteration and take track of the selection)
# params = all_RSS, all_R_squared, all_combs
# results = best_RSS, best_feature (results of the selection)
# detailed = parameter to see more detail about each single combination
def get_info_forwardS(ncomb,features,params,results,detailed):
    
    sepComb = "==="*30
    sepIter = "---"*30

    remaining, selected = features
    bestFeat, bestRSS = results
    
    print(f"{sepComb}\nIter n.{ncomb}:\n\
Choose {ncomb}-length combinations of the remaining variables\n\n\
Remaining features: {remaining}\n\
Features selected: {selected}")

    if detailed == 1:
        RSS, R_squared, Combs = params
        
        for niter in range(0,len(Combs)):
            var0 = Combs[niter]
            var1 = RSS[niter] 
            var2 = R_squared[niter]
            print(f"\nComb n.{niter+1}: {var0}\n\
{sepIter}\n\
RSS test: {var1}\n\
R_squared: {var2}\
")
        
    print(f"\nSelected variables: {bestFeat}\n\
min RSS: {bestRSS}\n\
{sepComb}\n")

    return


## Print information about an iteration backward subset selection with subset size ncomb
# ncomb = number of the iteration
# features = [remaining_features,dropped_features_list] (used to do the iteration and take track of the selection)
# params = [all_RSS,all_R_squared,all_combs]
# results = [best_RSS,dropped_feature]
# detailed = parameter to see more detail about each single combination
def get_info_backwardS(ncomb,features,params,results,detailed):
    
    sepComb = "==="*30
    sepIter = "---"*30

    remaining, dropped = features
    droppedFeat, bestRSS = results

    print(f"{sepComb}\nIter n.{8 - ncomb}:\n\n\
At the beginning we have:\n\
Remaining features: {remaining}\n\
Dropped features: {dropped}\n\n\
Now we compare the model selecting {ncomb} variables")

    if detailed == 1:
        RSS, R_squared, Combs = params
        
        for niter in range(0,len(Combs)):
            var0 = Combs[niter]
            var1 = RSS[niter] 
            var2 = R_squared[niter]
            print(f"\n\nComb n.{niter+1}: {var0}\n\
{sepIter}\n\
candidate dropped feature: {list(set(remaining)-set(var0))}\n\
RSS test: {var1}\
")

    print(f"\n\nAt the end we have:\n\
min RSS: {bestRSS}\n\
We drop: {droppedFeat}\n\
{sepComb}\n")
    
    return


## Print information about an iteration backward subset selection with subset size ncomb
# ncomb = number of the iteration
# features = [remaining_features,dropped_features_list] (used to do the iteration and take track of the selection)
# params = [all_RSS,all_R_squared,all_combs]
# results = [best_RSS,dropped_feature]
# detailed = parameter to see more detail about each single combination
def get_info_backwardS_Zscore(ncomb,features,params,results,detailed):
    
    sepComb = "==="*30
    sepIter = "---"*30

    remaining, dropped = features
    droppedFeat, bestRSS = results

    print(f"{sepComb}\nIter n.{8 - ncomb}:\n\n\
At the beginning we have:\n\
Remaining features: {remaining}\n\
Dropped features: {dropped}\n")

    print("\nThe Z-scores are:\n",Zscores)

    print(f"\n\nAt the end we have:\n\
min RSS: {bestRSS}\n\
We drop: {droppedFeat}\n\
{sepComb}\n")
    
    return
```

## **Best Subset Selection**

<h4>3. Starting from the `ols models` achieved in the last steps, perform best-subset selection.</h4>

* Generate one model for each combination of the 8 variables available
* For each model compute the RSS on training and test set, the number of variables and the \\(R^2\\) of the model
* Save these numbers in suitable data structures


```python
## range
variables = data.columns.tolist() # excluding 'const'

## Initialize the list where we temporarily store data
RSS_train_list, RSS_test_list, R_squared_list = [], [], []
numb_features, features_list = [], []

for k in range(1,len(variables) + 1):

#     niter = 0
#     print("---"*30,f"\nStart by choosing {k} variables\n")
    
    # Looping over all possible combinations of k variables
    for combo in itertools.combinations(variables,k):
        
#         niter = niter+1 
        
        # Compute all the statistics we need
        RSS_train, RSS_test, Rsquared_train = LinReg(X_train[list(combo)], Y_train, X_test[list(combo)], Y_test)
        
#         rnd = 4
#         print(f"{niter}. Variables: {list(combo)}\n\
#         RSS train: {RSS_train.round(rnd)}\n\
#         RSS test: {RSS_test.round(rnd)}\n\
#         R^2 train: {Rsquared_train.round(rnd)}\n")
        
        # Save the statistics
        RSS_train_list.append(RSS_train)
        RSS_test_list.append(RSS_test)
        R_squared_list.append(Rsquared_train)
        
        # Save features and number of features
        features_list.append(combo)
        numb_features.append(len(combo))   

#     print(f"\nUsing {k} variables we have computed {niter} models")
#     print("---"*30,"\n")

#Store in DataFrame
df_BestS = pd.DataFrame({'numb_features': numb_features,\
                         'RSS_train': RSS_train_list,\
                         'RSS_test': RSS_test_list,\
                         'R_squared': R_squared_list,\
                         'features': features_list})
```


```python
df_BestS
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_train</th>
      <th style='text-align:center; vertical-align:middle'>RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>R_squared</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>44.528583</td>
      <td style='text-align:center; vertical-align:middle'>14.392162</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol,)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>73.613540</td>
      <td style='text-align:center; vertical-align:middle'>30.402846</td>
      <td style='text-align:center; vertical-align:middle'>0.235434</td>
      <td style='text-align:center; vertical-align:middle'>(lweight,)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>91.292039</td>
      <td style='text-align:center; vertical-align:middle'>33.846748</td>
      <td style='text-align:center; vertical-align:middle'>0.051821</td>
      <td style='text-align:center; vertical-align:middle'>(age,)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>89.624912</td>
      <td style='text-align:center; vertical-align:middle'>35.298771</td>
      <td style='text-align:center; vertical-align:middle'>0.069136</td>
      <td style='text-align:center; vertical-align:middle'>(lbph,)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>66.422403</td>
      <td style='text-align:center; vertical-align:middle'>20.632078</td>
      <td style='text-align:center; vertical-align:middle'>0.310122</td>
      <td style='text-align:center; vertical-align:middle'>(svi,)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>...</th>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>250</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>31.570706</td>
      <td style='text-align:center; vertical-align:middle'>14.702112</td>
      <td style='text-align:center; vertical-align:middle'>0.672100</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, svi, lcp, gleason, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>251</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>30.414990</td>
      <td style='text-align:center; vertical-align:middle'>17.034552</td>
      <td style='text-align:center; vertical-align:middle'>0.684103</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, lcp, gleason, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>252</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>33.265433</td>
      <td style='text-align:center; vertical-align:middle'>16.754443</td>
      <td style='text-align:center; vertical-align:middle'>0.654498</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, age, lbph, svi, lcp, gleason, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>253</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>44.036622</td>
      <td style='text-align:center; vertical-align:middle'>22.633329</td>
      <td style='text-align:center; vertical-align:middle'>0.542626</td>
      <td style='text-align:center; vertical-align:middle'>(lweight, age, lbph, svi, lcp, gleason, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>254</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>29.426384</td>
      <td style='text-align:center; vertical-align:middle'>15.638220</td>
      <td style='text-align:center; vertical-align:middle'>0.694371</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, gleason...</td>
    </tr>
  </tbody>
</table>
<p>255 rows × 5 columns</p>
</div>


<h4>Find the best subset (of variables) for each number of features</h4>

Now our dataframe **`df_BestS`** has a row for each model computed, and it's not easy to handle. For this reason, we extract the best model for each number of variables by observing the interesting parameter.\
We consider as interesting parameter:
1. minimum RSS train
2. minimum RSS test
3. maximum \\(R^2\\)


```python
# Create new df, selection only best subsets of variables (based on RSS and R^2)
df_BestS_RSS_train= df_BestS[df_BestS.groupby('numb_features')['RSS_train'].transform(min) == df_BestS['RSS_train']]
df_BestS_RSS_test = df_BestS[df_BestS.groupby('numb_features')['RSS_test'].transform(min) == df_BestS['RSS_test']]
df_BestS_R_squared = df_BestS[df_BestS.groupby('numb_features')['R_squared'].transform(max) == df_BestS['R_squared']]
```


```python
df_BestS_RSS_train
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_train</th>
      <th style='text-align:center; vertical-align:middle'>RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>R_squared</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>44.528583</td>
      <td style='text-align:center; vertical-align:middle'>14.392162</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol,)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>8</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>37.091846</td>
      <td style='text-align:center; vertical-align:middle'>14.774470</td>
      <td style='text-align:center; vertical-align:middle'>0.614756</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>38</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>34.907749</td>
      <td style='text-align:center; vertical-align:middle'>12.015924</td>
      <td style='text-align:center; vertical-align:middle'>0.637441</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, svi)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>97</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>32.814995</td>
      <td style='text-align:center; vertical-align:middle'>13.689964</td>
      <td style='text-align:center; vertical-align:middle'>0.659176</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>174</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>32.069447</td>
      <td style='text-align:center; vertical-align:middle'>14.577726</td>
      <td style='text-align:center; vertical-align:middle'>0.666920</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>229</th>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>30.539778</td>
      <td style='text-align:center; vertical-align:middle'>16.457800</td>
      <td style='text-align:center; vertical-align:middle'>0.682807</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, lcp, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>247</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>29.437300</td>
      <td style='text-align:center; vertical-align:middle'>15.495405</td>
      <td style='text-align:center; vertical-align:middle'>0.694258</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>254</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>29.426384</td>
      <td style='text-align:center; vertical-align:middle'>15.638220</td>
      <td style='text-align:center; vertical-align:middle'>0.694371</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, gleason...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_BestS_RSS_test
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_train</th>
      <th style='text-align:center; vertical-align:middle'>RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>R_squared</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>44.528583</td>
      <td style='text-align:center; vertical-align:middle'>14.392162</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol,)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>11</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>42.312584</td>
      <td style='text-align:center; vertical-align:middle'>11.583584</td>
      <td style='text-align:center; vertical-align:middle'>0.560532</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, svi)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>52</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>42.267034</td>
      <td style='text-align:center; vertical-align:middle'>11.484038</td>
      <td style='text-align:center; vertical-align:middle'>0.561005</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, svi, gleason)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>112</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>42.223638</td>
      <td style='text-align:center; vertical-align:middle'>11.612573</td>
      <td style='text-align:center; vertical-align:middle'>0.561456</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, age, svi, gleason)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>167</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>34.170209</td>
      <td style='text-align:center; vertical-align:middle'>11.497692</td>
      <td style='text-align:center; vertical-align:middle'>0.645101</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, svi, gleason)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>226</th>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>33.642783</td>
      <td style='text-align:center; vertical-align:middle'>12.009380</td>
      <td style='text-align:center; vertical-align:middle'>0.650579</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, svi, gleason, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>246</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>30.958630</td>
      <td style='text-align:center; vertical-align:middle'>13.492898</td>
      <td style='text-align:center; vertical-align:middle'>0.678457</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, gleason)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>254</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>29.426384</td>
      <td style='text-align:center; vertical-align:middle'>15.638220</td>
      <td style='text-align:center; vertical-align:middle'>0.694371</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, gleason...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# the same as selecting min RSS on training set
df_BestS_R_squared
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_train</th>
      <th style='text-align:center; vertical-align:middle'>RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>R_squared</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>44.528583</td>
      <td style='text-align:center; vertical-align:middle'>14.392162</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol,)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>8</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>37.091846</td>
      <td style='text-align:center; vertical-align:middle'>14.774470</td>
      <td style='text-align:center; vertical-align:middle'>0.614756</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>38</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>34.907749</td>
      <td style='text-align:center; vertical-align:middle'>12.015924</td>
      <td style='text-align:center; vertical-align:middle'>0.637441</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, svi)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>97</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>32.814995</td>
      <td style='text-align:center; vertical-align:middle'>13.689964</td>
      <td style='text-align:center; vertical-align:middle'>0.659176</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>174</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>32.069447</td>
      <td style='text-align:center; vertical-align:middle'>14.577726</td>
      <td style='text-align:center; vertical-align:middle'>0.666920</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>229</th>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>30.539778</td>
      <td style='text-align:center; vertical-align:middle'>16.457800</td>
      <td style='text-align:center; vertical-align:middle'>0.682807</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, lcp, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>247</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>29.437300</td>
      <td style='text-align:center; vertical-align:middle'>15.495405</td>
      <td style='text-align:center; vertical-align:middle'>0.694258</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, pgg45)</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>254</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>29.426384</td>
      <td style='text-align:center; vertical-align:middle'>15.638220</td>
      <td style='text-align:center; vertical-align:middle'>0.694371</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, gleason...</td>
    </tr>
  </tbody>
</table>
</div>



### **Plots for Best Selection**

<h4>4. + 5. + 6.  Generate some charts:</h4>

- x-axis: the subset size
- y-axis:
    * 4. RSS for training set of all the models generated at step 3
    * 5. \\(R^2\\) of all the models generated at step 3
    * 6. RSS for test set of all the models generated at step 3


```python
# Initialize the figure
width = 6
height = 6
nfig = 3
fig = plt.figure(figsize = (width*nfig,height))

# 1. RSS Training set plot
tmp_df1 = df_BestS;           # scatter plot
tmp_df2 = df_BestS_RSS_train; # plot the line of best values
ax1 = fig.add_subplot(1, nfig, 1)
ax1.scatter(tmp_df1.numb_features,tmp_df1.RSS_train, alpha = .2, color = 'darkblue');
ax1.set_xlabel('Subset Size k',fontsize=14);
ax1.set_ylabel('RSS',fontsize=14);
ax1.set_title('RSS on training set',fontsize=18);
ax1.plot(tmp_df2.numb_features,tmp_df2.RSS_train,color = 'r', label = 'Best subset'); # line of best values
ax1.grid(color='grey', linestyle='-', linewidth=0.5);
ax1.legend();

# 2. RSS Test set plot
tmp_df1 = df_BestS;           # scatter plot
tmp_df2 = df_BestS_RSS_test;  # plot the line of best values
ax2 = fig.add_subplot(1, nfig, 2);
ax2.scatter(tmp_df1.numb_features,tmp_df1.RSS_test, alpha = .2, color = 'darkblue');
ax2.set_xlabel('Subset Size k',fontsize=14);
ax2.set_ylabel('RSS',fontsize=14);
ax2.set_title('RSS on test set',fontsize=18);
ax2.plot(tmp_df2.numb_features,tmp_df2.RSS_test,color = 'r', label = 'Best subset'); # line of best values
ax2.grid(color='grey', linestyle='-', linewidth=0.5);
ax2.legend();

# 3. R^2 plot
tmp_df1 = df_BestS;           # scatter plot
tmp_df2 = df_BestS_R_squared;  # plot the line of best values
ax3 = fig.add_subplot(1, nfig, 3);
ax3.scatter(tmp_df1.numb_features,tmp_df1.R_squared, alpha = .2, color = 'darkblue');
ax3.set_xlabel('Subset Size k',fontsize=14);
ax3.set_ylabel('$R^2$',fontsize=14);
ax3.set_ylim(bottom=-0.1,top=1.1)
ax3.set_title('$R^2$ on training set',fontsize=18);
ax3.plot(tmp_df2.numb_features,tmp_df2.R_squared,color = 'r', label = 'Best subset'); # line of best values
ax3.grid(color='grey', linestyle='-', linewidth=0.5);
ax3.legend();

fig.suptitle('Best Subset Selection',fontsize=25, y=0.98);
fig.subplots_adjust(top=0.8)
plt.show();
```


![png](/posts/sl-ex3-subsetselection/output_27_0.png)



### **Estimating test error by adjusting training error**

In general, **the training set MSE underestimates the test MSE**. In fact we fit a model to the training data using least squares and we estimate the coefficients of the regression in such a way the training RSS is minimized. So the training RSS decreases as we add more variables to the model, but the test RSS may not!

For this reason the training RSS (and so also  \\(R\^2\\)) may not be used directly for selecting the best model, but we need to adjust them to get an estimate of the test error.

<h4>Attempt n.1: using sklearn library</h4>


```python
AIC_list, BIC_list, R_squared_adj_list =[], [], [] # from sklearn I can get no Cp estimate out of the box.
best_features = df_BestS_RSS_train['features']

for features in best_features:
    
    regr = sm.OLS(Y_train, sm.add_constant(X_train[list(features)])).fit()
        
    AIC_list.append(regr.aic)
    BIC_list.append(regr.bic)
    R_squared_adj_list.append(regr.rsquared_adj)
```


```python
#Store in DataFrame
df1 = pd.DataFrame({'numb_features': df_BestS_RSS_train['numb_features'],\
                    'RSS_train': df_BestS_RSS_train['RSS_train'],\
                    'features': df_BestS_RSS_train['features'],\
                    'AIC': AIC_list,\
                   'BIC': BIC_list,\
                   'R_squared_adj': R_squared_adj_list})

df1
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_train</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
      <th style='text-align:center; vertical-align:middle'>AIC</th>
      <th style='text-align:center; vertical-align:middle'>BIC</th>
      <th style='text-align:center; vertical-align:middle'>R_squared_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>44.528583</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol,)</td>
      <td style='text-align:center; vertical-align:middle'>166.764154</td>
      <td style='text-align:center; vertical-align:middle'>171.173540</td>
      <td style='text-align:center; vertical-align:middle'>0.530401</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>8</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>37.091846</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight)</td>
      <td style='text-align:center; vertical-align:middle'>156.520967</td>
      <td style='text-align:center; vertical-align:middle'>163.135045</td>
      <td style='text-align:center; vertical-align:middle'>0.602717</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>38</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>34.907749</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, svi)</td>
      <td style='text-align:center; vertical-align:middle'>154.454850</td>
      <td style='text-align:center; vertical-align:middle'>163.273620</td>
      <td style='text-align:center; vertical-align:middle'>0.620176</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>97</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>32.814995</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi)</td>
      <td style='text-align:center; vertical-align:middle'>152.312691</td>
      <td style='text-align:center; vertical-align:middle'>163.336154</td>
      <td style='text-align:center; vertical-align:middle'>0.637188</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>174</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>32.069447</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, pgg45)</td>
      <td style='text-align:center; vertical-align:middle'>152.772911</td>
      <td style='text-align:center; vertical-align:middle'>166.001067</td>
      <td style='text-align:center; vertical-align:middle'>0.639618</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>229</th>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>30.539778</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, lcp, pgg45)</td>
      <td style='text-align:center; vertical-align:middle'>151.498370</td>
      <td style='text-align:center; vertical-align:middle'>166.931219</td>
      <td style='text-align:center; vertical-align:middle'>0.651088</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>247</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>29.437300</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, pgg45)</td>
      <td style='text-align:center; vertical-align:middle'>151.034951</td>
      <td style='text-align:center; vertical-align:middle'>168.672492</td>
      <td style='text-align:center; vertical-align:middle'>0.657983</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>254</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>29.426384</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, gleason...</td>
      <td style='text-align:center; vertical-align:middle'>153.010102</td>
      <td style='text-align:center; vertical-align:middle'>172.852336</td>
      <td style='text-align:center; vertical-align:middle'>0.652215</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Initialize the figure
width = 6
height = 6
nfig = 3
fig = plt.figure(figsize = (width*nfig,height))

# 1. AIC
ax1 = fig.add_subplot(1, nfig, 1)
ax1.scatter(df1['numb_features'],df1['AIC'], alpha = .5, color = 'darkblue');
ax1.plot(df1['numb_features'],df1['AIC'],color = 'r', alpha = .5); # line of best values
ax1.plot(df1['AIC'].argmin()+1, df1['AIC'].min(), marker='x', markersize=20, color = 'b'); # best val
ax1.set_xlabel('Number of predictors',fontsize=14);
ax1.set_ylabel('AIC',fontsize=14);
ax1.grid(color='grey', linestyle='-', linewidth=0.5);

# 2. BIC
ax2 = fig.add_subplot(1, nfig, 2)
ax2.scatter(df1['numb_features'],df1['BIC'], alpha = .5, color = 'darkblue');
ax2.plot(df1['numb_features'],df1['BIC'],color = 'r', alpha = .5); # line of best values
ax2.plot(df1['BIC'].argmin()+1, df1['BIC'].min(), marker='x', markersize=20, color = 'b'); # best val
ax2.set_xlabel('Number of predictors',fontsize=14);
ax2.set_ylabel('BIC',fontsize=14);
ax2.grid(color='grey', linestyle='-', linewidth=0.5);

# 3. R2_adj
ax3 = fig.add_subplot(1, nfig, 3)
ax3.scatter(df1['numb_features'],df1['R_squared_adj'], alpha = .5, color = 'darkblue');
ax3.plot(df1['numb_features'],df1['R_squared_adj'],color = 'r', alpha = .5); # line of best values
ax3.plot(df1['R_squared_adj'].argmax()+1, df1['R_squared_adj'].max(), marker='x', markersize=20, color = 'b'); # best val
ax3.set_xlabel('Number of predictors',fontsize=14);
ax3.set_ylabel(r'$R^2$ adj',fontsize=14);
ax3.grid(color='grey', linestyle='-', linewidth=0.5);

fig.suptitle('Best Subset Selection:\n Estimating test error by adjusting training error',fontsize=25, y=0.98);
fig.subplots_adjust(top=0.8)
fig.subplots_adjust(wspace=0.275) # the amount of width reserved for blank space between subplots
plt.show();
```


![png](/posts/sl-ex3-subsetselection/output_32_0.png)


<h4>Attempt n.2: estimating AIC, BIC and R^2 adjusted by our own</h4>


```python
#Initializing useful variables
n = len(Y_train)
d = df_BestS_RSS_train['numb_features']
p = 8
RSS = df_BestS_RSS_train['RSS_train']
Rsquared = df_BestS_RSS_train['R_squared']

# Estimation of sigma^2: RSE of the general multiple Linear regression with p features
RSE = np.sqrt(min(RSS)/(n - p -1))  # min(RSS) is the RSS of the full model with p predictors
hat_sigma_squared = RSE**2

#Computing
AIC = (1/(n*hat_sigma_squared)) * (RSS + 2 * d * hat_sigma_squared )
BIC = (1/(n*hat_sigma_squared)) * (RSS +  np.log(n) * d * hat_sigma_squared )

ratio = 1 - Rsquared # RSS/TSS
R_squared_adj = 1 - ratio*(n-1)/(n-d-1)
```


```python
#Store in DataFrame
df2 = pd.DataFrame({'numb_features': df_BestS_RSS_train['numb_features'],\
                    'RSS_train': df_BestS_RSS_train['RSS_train'],\
                    'features': df_BestS_RSS_train['features'],\
                    'AIC': AIC,\
                    'BIC': BIC,\
                    'R_squared_adj': R_squared_adj})

df2
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_train</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
      <th style='text-align:center; vertical-align:middle'>AIC</th>
      <th style='text-align:center; vertical-align:middle'>BIC</th>
      <th style='text-align:center; vertical-align:middle'>R_squared_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>44.528583</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol,)</td>
      <td style='text-align:center; vertical-align:middle'>1.339802</td>
      <td style='text-align:center; vertical-align:middle'>1.372708</td>
      <td style='text-align:center; vertical-align:middle'>0.530401</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>8</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>37.091846</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight)</td>
      <td style='text-align:center; vertical-align:middle'>1.150877</td>
      <td style='text-align:center; vertical-align:middle'>1.216689</td>
      <td style='text-align:center; vertical-align:middle'>0.602717</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>38</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>34.907749</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, svi)</td>
      <td style='text-align:center; vertical-align:middle'>1.116476</td>
      <td style='text-align:center; vertical-align:middle'>1.215193</td>
      <td style='text-align:center; vertical-align:middle'>0.620176</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>97</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>32.814995</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi)</td>
      <td style='text-align:center; vertical-align:middle'>1.084761</td>
      <td style='text-align:center; vertical-align:middle'>1.216385</td>
      <td style='text-align:center; vertical-align:middle'>0.637188</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>174</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>32.069447</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, pgg45)</td>
      <td style='text-align:center; vertical-align:middle'>1.092680</td>
      <td style='text-align:center; vertical-align:middle'>1.257209</td>
      <td style='text-align:center; vertical-align:middle'>0.639618</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>229</th>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>30.539778</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, lbph, svi, lcp, pgg45)</td>
      <td style='text-align:center; vertical-align:middle'>1.077530</td>
      <td style='text-align:center; vertical-align:middle'>1.274965</td>
      <td style='text-align:center; vertical-align:middle'>0.651088</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>247</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>29.437300</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, pgg45)</td>
      <td style='text-align:center; vertical-align:middle'>1.074948</td>
      <td style='text-align:center; vertical-align:middle'>1.305289</td>
      <td style='text-align:center; vertical-align:middle'>0.657983</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>254</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>29.426384</td>
      <td style='text-align:center; vertical-align:middle'>(lcavol, lweight, age, lbph, svi, lcp, gleason...</td>
      <td style='text-align:center; vertical-align:middle'>1.104478</td>
      <td style='text-align:center; vertical-align:middle'>1.367724</td>
      <td style='text-align:center; vertical-align:middle'>0.652215</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Initialize the figure
width = 6
height = 6
nfig = 3
fig = plt.figure(figsize = (width*nfig,height))

# 1. AIC
ax1 = fig.add_subplot(1, nfig, 1)
ax1.scatter(df2['numb_features'],df2['AIC'], alpha = .5, color = 'darkblue');
ax1.plot(df2['numb_features'],df2['AIC'],color = 'r', alpha = .5); # line of best values
ax1.plot(df2['AIC'].argmin()+1, df2['AIC'].min(), marker='x', markersize=20, color = 'b'); # best val
ax1.set_xlabel('Number of predictors',fontsize=14);
ax1.set_ylabel('AIC',fontsize=14);
ax1.grid(color='grey', linestyle='-', linewidth=0.5);

# 2. BIC
ax2 = fig.add_subplot(1, nfig, 2)
ax2.scatter(df2['numb_features'],df2['BIC'], alpha = .5, color = 'darkblue');
ax2.plot(df2['numb_features'],df2['BIC'],color = 'r', alpha = .5); # line of best values
ax2.plot(df2['BIC'].argmin()+1, df2['BIC'].min(), marker='x', markersize=20, color = 'b'); # best val
ax2.set_xlabel('Number of predictors',fontsize=14);
ax2.set_ylabel('BIC',fontsize=14);
ax2.grid(color='grey', linestyle='-', linewidth=0.5);

# 3. R2_adj
ax3 = fig.add_subplot(1, nfig, 3)
ax3.scatter(df2['numb_features'],df2['R_squared_adj'], alpha = .5, color = 'darkblue');
ax3.plot(df2['numb_features'],df2['R_squared_adj'],color = 'r', alpha = .5); # line of best values
ax3.plot(df2['R_squared_adj'].argmax()+1, df2['R_squared_adj'].max(), marker='x', markersize=20, color = 'b'); # best val
ax3.set_xlabel('Number of predictors',fontsize=14);
ax3.set_ylabel(r'$R^2$ adj',fontsize=14);
ax3.grid(color='grey', linestyle='-', linewidth=0.5);

fig.suptitle('Best Subset Selection:\n Estimating test error by adjusting training error',fontsize=25, y=0.98);
fig.subplots_adjust(top=0.8)
fig.subplots_adjust(wspace=0.275) # the amount of width reserved for blank space between subplots
plt.show();
```


![png](/posts/sl-ex3-subsetselection/output_36_0.png)

## **Forward selection**

For computational reasons, the best subset cannot be applied for any large n due to the \\(2^n\\) complexity.\
Forward Stepwise begins with a model containing no predictors, and then adds predictors to the model, one at the time. At each step, the variable that gives the greatest additional improvement to the fit is added to the model.

<h4>7. Perform forward selection</h4>

* Start from the empty model
* Add at each step the variable that **minimizes the RSS computed on test set** (other performance measures can be used)


```python
# to print some info:
flag = 1
detailed = 1
```

```python
## range
variables = data.columns.tolist()
remaining_features, selected_features = variables.copy(), []

## Initialize the list where we temporarily store data
RSS_test_list, min_RSS_test_list, R_squared_list = [], [], []
numb_features, features_list = [], []

# Loop over the number of variables
for k in range(1,len(variables)+1):
    
    # store some info for each k
    all_RSS, all_R_squared, all_combs = [],[], []
    best_RSS = np.inf # initialize the best RSS as +inf

    # choose one variable in the remaining features
    for var in remaining_features:
        
        tmpComb = selected_features + [var]; # combination of variables
        
        # Compute all the statistics we need
        _, RSS_test, R_squared = LinReg(X_train[tmpComb], Y_train, X_test[tmpComb], Y_test) # we don't want RSS on training set
        
        # save temporary stats
        all_RSS.append(RSS_test)
        all_R_squared.append(R_squared)
        all_combs.append(tmpComb)

        # update if we reach a better RSS
        if RSS_test < best_RSS:
            best_RSS = RSS_test
            best_R_squared = R_squared
            best_feature = var
   
    # Print some information, before upgrading the features
    if flag == 1:
        features = [remaining_features,selected_features]
        params = [all_RSS,all_R_squared,all_combs]
        results = [best_feature,best_RSS]
        get_info_forwardS(k,features,params,results,detailed)
        
    # Save the statistics
    RSS_test_list.append(all_RSS)
    min_RSS_test_list.append(best_RSS)
    R_squared_list.append(best_R_squared)
    
    # Update variables for next loop
    selected_features.append(best_feature)
    remaining_features.remove(best_feature)

    # Save features and number of features
    features_list.append(selected_features.copy())
    numb_features.append(len(selected_features))
    
# Store in DataFrame
df_ForwardS = pd.DataFrame({'numb_features': numb_features,\
                            'RSS_test' : RSS_test_list,\
                            'min_RSS_test': min_RSS_test_list,\
                            'R_squared': R_squared_list,\
                            'features': features_list})
```

    ==========================================================================================
    Iter n.1:
    Choose 1-length combinations of the remaining variables
    
    Remaining features: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    Features selected: []
    
    Comb n.1: ['lcavol']
    ------------------------------------------------------------------------------------------
    RSS test: 14.392161587304827
    R_squared: 0.5375164690552883
    
    Comb n.2: ['lweight']
    ------------------------------------------------------------------------------------------
    RSS test: 30.402845602615997
    R_squared: 0.23543378299009432
    
    Comb n.3: ['age']
    ------------------------------------------------------------------------------------------
    RSS test: 33.846748424133
    R_squared: 0.05182105437299367
    
    Comb n.4: ['lbph']
    ------------------------------------------------------------------------------------------
    RSS test: 35.29877101280492
    R_squared: 0.06913619684911343
    
    Comb n.5: ['svi']
    ------------------------------------------------------------------------------------------
    RSS test: 20.632078139876853
    R_squared: 0.3101224985902339
    
    Comb n.6: ['lcp']
    ------------------------------------------------------------------------------------------
    RSS test: 16.34576112489144
    R_squared: 0.23931977441332264
    
    Comb n.7: ['gleason']
    ------------------------------------------------------------------------------------------
    RSS test: 25.529830407597565
    R_squared: 0.11725680432657692
    
    Comb n.8: ['pgg45']
    ------------------------------------------------------------------------------------------
    RSS test: 28.61697167270323
    R_squared: 0.20074696985742568
    
    Selected variables: lcavol
    min RSS: 14.392161587304827
    ==========================================================================================
    
    ==========================================================================================
    Iter n.2:
    Choose 2-length combinations of the remaining variables
    
    Remaining features: ['lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    Features selected: ['lcavol']
    
    Comb n.1: ['lcavol', 'lweight']
    ------------------------------------------------------------------------------------------
    RSS test: 14.77447043041511
    R_squared: 0.614756035022443
    
    Comb n.2: ['lcavol', 'age']
    ------------------------------------------------------------------------------------------
    RSS test: 14.454316401468965
    R_squared: 0.53785859616379
    
    Comb n.3: ['lcavol', 'lbph']
    ------------------------------------------------------------------------------------------
    RSS test: 16.237234684619743
    R_squared: 0.5846312407995875
    
    Comb n.4: ['lcavol', 'svi']
    ------------------------------------------------------------------------------------------
    RSS test: 11.58358360007023
    R_squared: 0.5605323092792945
    
    Comb n.5: ['lcavol', 'lcp']
    ------------------------------------------------------------------------------------------
    RSS test: 15.038931408454383
    R_squared: 0.5381501805446671
    
    Comb n.6: ['lcavol', 'gleason']
    ------------------------------------------------------------------------------------------
    RSS test: 14.140851461764317
    R_squared: 0.5386018755651162
    
    Comb n.7: ['lcavol', 'pgg45']
    ------------------------------------------------------------------------------------------
    RSS test: 13.827662568490943
    R_squared: 0.5489982126930548
    
    Selected variables: svi
    min RSS: 11.58358360007023
    ==========================================================================================
    
    ==========================================================================================
    Iter n.3:
    Choose 3-length combinations of the remaining variables
    
    Remaining features: ['lweight', 'age', 'lbph', 'lcp', 'gleason', 'pgg45']
    Features selected: ['lcavol', 'svi']
    
    Comb n.1: ['lcavol', 'svi', 'lweight']
    ------------------------------------------------------------------------------------------
    RSS test: 12.015924403078804
    R_squared: 0.6374405385171893
    
    Comb n.2: ['lcavol', 'svi', 'age']
    ------------------------------------------------------------------------------------------
    RSS test: 11.71416952499475
    R_squared: 0.5612383365966047
    
    Comb n.3: ['lcavol', 'svi', 'lbph']
    ------------------------------------------------------------------------------------------
    RSS test: 14.353733928669755
    R_squared: 0.6264131675413883
    
    Comb n.4: ['lcavol', 'svi', 'lcp']
    ------------------------------------------------------------------------------------------
    RSS test: 13.285116624863331
    R_squared: 0.5714253936028364
    
    Comb n.5: ['lcavol', 'svi', 'gleason']
    ------------------------------------------------------------------------------------------
    RSS test: 11.484037587414818
    R_squared: 0.5610054092768177
    
    Comb n.6: ['lcavol', 'svi', 'pgg45']
    ------------------------------------------------------------------------------------------
    RSS test: 11.632246428034497
    R_squared: 0.5651377944544718
    
    Selected variables: gleason
    min RSS: 11.484037587414818
    ==========================================================================================
    
    ==========================================================================================
    Iter n.4:
    Choose 4-length combinations of the remaining variables
    
    Remaining features: ['lweight', 'age', 'lbph', 'lcp', 'pgg45']
    Features selected: ['lcavol', 'svi', 'gleason']
    
    Comb n.1: ['lcavol', 'svi', 'gleason', 'lweight']
    ------------------------------------------------------------------------------------------
    RSS test: 11.95602065641986
    R_squared: 0.6405674126734184
    
    Comb n.2: ['lcavol', 'svi', 'gleason', 'age']
    ------------------------------------------------------------------------------------------
    RSS test: 11.612572746859495
    R_squared: 0.5614561297716232
    
    Comb n.3: ['lcavol', 'svi', 'gleason', 'lbph']
    ------------------------------------------------------------------------------------------
    RSS test: 14.355178421938225
    R_squared: 0.6266591566646609
    
    Comb n.4: ['lcavol', 'svi', 'gleason', 'lcp']
    ------------------------------------------------------------------------------------------
    RSS test: 13.185630584098526
    R_squared: 0.5741698339447385
    
    Comb n.5: ['lcavol', 'svi', 'gleason', 'pgg45']
    ------------------------------------------------------------------------------------------
    RSS test: 11.919697961636555
    R_squared: 0.5664871741058846
    
    Selected variables: age
    min RSS: 11.612572746859495
    ==========================================================================================
    
    ==========================================================================================
    Iter n.5:
    Choose 5-length combinations of the remaining variables
    
    Remaining features: ['lweight', 'lbph', 'lcp', 'pgg45']
    Features selected: ['lcavol', 'svi', 'gleason', 'age']
    
    Comb n.1: ['lcavol', 'svi', 'gleason', 'age', 'lweight']
    ------------------------------------------------------------------------------------------
    RSS test: 11.497691985782547
    R_squared: 0.64510079253458
    
    Comb n.2: ['lcavol', 'svi', 'gleason', 'age', 'lbph']
    ------------------------------------------------------------------------------------------
    RSS test: 13.947298386201728
    R_squared: 0.6294946547801643
    
    Comb n.3: ['lcavol', 'svi', 'gleason', 'age', 'lcp']
    ------------------------------------------------------------------------------------------
    RSS test: 13.234367776364218
    R_squared: 0.574264491209014
    
    Comb n.4: ['lcavol', 'svi', 'gleason', 'age', 'pgg45']
    ------------------------------------------------------------------------------------------
    RSS test: 12.141763999596828
    R_squared: 0.5670143538267589
    
    Selected variables: lweight
    min RSS: 11.497691985782547
    ==========================================================================================
    
    ==========================================================================================
    Iter n.6:
    Choose 6-length combinations of the remaining variables
    
    Remaining features: ['lbph', 'lcp', 'pgg45']
    Features selected: ['lcavol', 'svi', 'gleason', 'age', 'lweight']
    
    Comb n.1: ['lcavol', 'svi', 'gleason', 'age', 'lweight', 'lbph']
    ------------------------------------------------------------------------------------------
    RSS test: 12.98960324749752
    R_squared: 0.6696630706385877
    
    Comb n.2: ['lcavol', 'svi', 'gleason', 'age', 'lweight', 'lcp']
    ------------------------------------------------------------------------------------------
    RSS test: 12.690514046312407
    R_squared: 0.6563534839394491
    
    Comb n.3: ['lcavol', 'svi', 'gleason', 'age', 'lweight', 'pgg45']
    ------------------------------------------------------------------------------------------
    RSS test: 12.009380388381148
    R_squared: 0.6505787510646934
    
    Selected variables: pgg45
    min RSS: 12.009380388381148
    ==========================================================================================
    
    ==========================================================================================
    Iter n.7:
    Choose 7-length combinations of the remaining variables
    
    Remaining features: ['lbph', 'lcp']
    Features selected: ['lcavol', 'svi', 'gleason', 'age', 'lweight', 'pgg45']
    
    Comb n.1: ['lcavol', 'svi', 'gleason', 'age', 'lweight', 'pgg45', 'lbph']
    ------------------------------------------------------------------------------------------
    RSS test: 13.834231490831279
    R_squared: 0.6760051914465672
    
    Comb n.2: ['lcavol', 'svi', 'gleason', 'age', 'lweight', 'pgg45', 'lcp']
    ------------------------------------------------------------------------------------------
    RSS test: 14.702111705571856
    R_squared: 0.6720997902395782
    
    Selected variables: lbph
    min RSS: 13.834231490831279
    ==========================================================================================
    
    ==========================================================================================
    Iter n.8:
    Choose 8-length combinations of the remaining variables
    
    Remaining features: ['lcp']
    Features selected: ['lcavol', 'svi', 'gleason', 'age', 'lweight', 'pgg45', 'lbph']
    
    Comb n.1: ['lcavol', 'svi', 'gleason', 'age', 'lweight', 'pgg45', 'lbph', 'lcp']
    ------------------------------------------------------------------------------------------
    RSS test: 15.638220165228002
    R_squared: 0.6943711796768238
    
    Selected variables: lcp
    min RSS: 15.638220165228002
    ==========================================================================================
    


```python
# look at the result
df_ForwardS
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>min_RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>R_squared</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>[14.392161587304827, 30.402845602615997, 33.84...</td>
      <td style='text-align:center; vertical-align:middle'>14.392162</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>[14.77447043041511, 14.454316401468965, 16.237...</td>
      <td style='text-align:center; vertical-align:middle'>11.583584</td>
      <td style='text-align:center; vertical-align:middle'>0.560532</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>[12.015924403078804, 11.71416952499475, 14.353...</td>
      <td style='text-align:center; vertical-align:middle'>11.484038</td>
      <td style='text-align:center; vertical-align:middle'>0.561005</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi, gleason]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>[11.95602065641986, 11.612572746859495, 14.355...</td>
      <td style='text-align:center; vertical-align:middle'>11.612573</td>
      <td style='text-align:center; vertical-align:middle'>0.561456</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi, gleason, age]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>[11.497691985782547, 13.947298386201728, 13.23...</td>
      <td style='text-align:center; vertical-align:middle'>11.497692</td>
      <td style='text-align:center; vertical-align:middle'>0.645101</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi, gleason, age, lweight]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>5</th>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>[12.98960324749752, 12.690514046312407, 12.009...</td>
      <td style='text-align:center; vertical-align:middle'>12.009380</td>
      <td style='text-align:center; vertical-align:middle'>0.650579</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi, gleason, age, lweight, pgg45]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>6</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>[13.834231490831279, 14.702111705571856]</td>
      <td style='text-align:center; vertical-align:middle'>13.834231</td>
      <td style='text-align:center; vertical-align:middle'>0.676005</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi, gleason, age, lweight, pgg45, lbph]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>[15.638220165228002]</td>
      <td style='text-align:center; vertical-align:middle'>15.638220</td>
      <td style='text-align:center; vertical-align:middle'>0.694371</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi, gleason, age, lweight, pgg45, lb...</td>
    </tr>
  </tbody>
</table>
</div>



### **Plot for Forward Selection**

<h4>8. Generate a chart having:</h4>

    - x-axis: the subset size
    - y-axis: the RSS for the test set of the models generated at step 7


```python
# Initialize the figure
width = 6
height = 6
nfig = 1
fig = plt.figure(figsize = (width*nfig,height))

# 1. RSS Test set plot
tmp_df = df_ForwardS;
ax = fig.add_subplot(1, nfig, 1)
for i in range(0,len(tmp_df.RSS_test)):
    ax.scatter([tmp_df.numb_features[i]]*(len(tmp_df.RSS_test[i])),tmp_df.RSS_test[i], alpha = .2, color = 'darkblue');
    
ax.set_xlabel('Subset Size k',fontsize=14);
ax.set_ylabel('RSS',fontsize=14);
ax.set_title('RSS on test set',fontsize=18);
ax.plot(tmp_df.numb_features,tmp_df.min_RSS_test,color = 'r', label = 'Best subset'); # line of best values
ax.grid(color='grey', linestyle='-', linewidth=0.5);
ax.legend();

fig.suptitle('Forward Subset Selection',fontsize=25, y=0.98);
fig.subplots_adjust(top=0.8)
plt.show()
```


![png](/posts/sl-ex3-subsetselection/output_37_0.png)


## **Backward selection**

Another alternative to best subset selection is Backward Stepwise Selection.\
Backward Stepwise begins with a model containing all the predictors, and then removes predictors to the model, one at the time. At each step, the variable that gives the least improvement to the fit is removed to the model.

Backward selection requires that the **number of samples \\(n\\) is larger than the number of variables \\(p\\)**. Instead, Forward selection can be used evene \\(n<p\\).

<h4>9. Perform backward selection</h4>

* Start from the full model
* Remove at each step the variable that minimizes the RSS (other performance measures can be used)


```python
# a flag to print some info, values {0,1}
flag = 1     # for short info
detailed = 1
```


```python
## range
variables = data.columns.tolist()         # excluding 'const'
remaining_features, dropped_features_list = variables.copy(), []

## Initialize the list where we temporarily store data
RSS_test_list, min_RSS_test_list, R_squared_list = [], [], []
numb_features, features_list = [], []

# run over the number of variables
for k in range(len(variables),0,-1):

    # initialization
    best_RSS = np.inf
    all_RSS, all_R_squared, all_combs = [],[], []

    for combo in itertools.combinations(remaining_features,k):

        # Compute the stats we need
        tmpComb = list(combo)
        _, RSS_test, R_squared = LinReg(X_train[tmpComb], Y_train, X_test[tmpComb], Y_test) # we don't want RSS on training set
        
        # store all the RSS
        all_RSS.append(RSS_test)
        all_R_squared.append(R_squared)
        all_combs.append(tmpComb)
        
        if RSS_test < best_RSS:
            best_RSS = RSS_test
            best_R_squared = R_squared
            dropped_list = list(set(remaining_features)-set(tmpComb))
 
    # Print some information, before upgrading the features
    if flag == 1:
        features = [remaining_features,dropped_features_list]
        params = [all_RSS,all_R_squared,all_combs]
        if dropped_list: # only if dropped_feature is not an empty list
            dropped_feature = dropped_list[0]
            results = [dropped_feature,best_RSS]
        else:
            results = [[],best_RSS]
            
        get_info_backwardS(k,features,params,results,detailed)
        
    # Updating variables for next loop
    if dropped_list: # only if dropped_feature is not an empty list
        dropped_feature = dropped_list[0]
        remaining_features.remove(dropped_feature)
        dropped_features_list.append(dropped_feature)
    else:
        dropped_features_list.append([]) # at the initial iteration we drop nothing!
        
    # Save stats
    min_RSS_test_list.append(best_RSS)
    RSS_test_list.append(all_RSS.copy())
    R_squared_list.append(best_R_squared.copy())

    # Save features and number of features
    numb_features.append(len(remaining_features))
    features_list.append(remaining_features.copy())
    
# Store in DataFrame
df_BackwardS = pd.DataFrame({'numb_features': numb_features,\
                             'RSS_test' : RSS_test_list,\
                             'min_RSS_test': min_RSS_test_list,\
                             'R_squared': R_squared_list,\
                             'dropped_feature': dropped_features_list,\
                             'features': features_list})
```

    ==========================================================================================
    Iter n.0:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    Dropped features: []
    
    Now we compare the model selecting 8 variables
    
    
    Comb n.1: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: []
    RSS test: 15.638220165228002
    
    
    At the end we have:
    min RSS: 15.638220165228002
    We drop: []
    ==========================================================================================
    
    ==========================================================================================
    Iter n.1:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    Dropped features: [[]]
    
    Now we compare the model selecting 7 variables
    
    
    Comb n.1: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['pgg45']
    RSS test: 13.492898446056923
    
    
    Comb n.2: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'pgg45']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['gleason']
    RSS test: 15.495404626758
    
    
    Comb n.3: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'gleason', 'pgg45']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcp']
    RSS test: 13.83423149083128
    
    
    Comb n.4: ['lcavol', 'lweight', 'age', 'lbph', 'lcp', 'gleason', 'pgg45']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['svi']
    RSS test: 17.516627850269806
    
    
    Comb n.5: ['lcavol', 'lweight', 'age', 'svi', 'lcp', 'gleason', 'pgg45']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lbph']
    RSS test: 14.702111705571852
    
    
    Comb n.6: ['lcavol', 'lweight', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['age']
    RSS test: 17.03455209459629
    
    
    Comb n.7: ['lcavol', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lweight']
    RSS test: 16.754443499511755
    
    
    Comb n.8: ['lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcavol']
    RSS test: 22.63332934025175
    
    
    At the end we have:
    min RSS: 13.492898446056923
    We drop: pgg45
    ==========================================================================================
    
    ==========================================================================================
    Iter n.2:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason']
    Dropped features: [[], 'pgg45']
    
    Now we compare the model selecting 6 variables
    
    
    Comb n.1: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['gleason']
    RSS test: 13.43573406046562
    
    
    Comb n.2: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcp']
    RSS test: 12.98960324749752
    
    
    Comb n.3: ['lcavol', 'lweight', 'age', 'lbph', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['svi']
    RSS test: 15.244637591703853
    
    
    Comb n.4: ['lcavol', 'lweight', 'age', 'svi', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lbph']
    RSS test: 12.690514046312394
    
    
    Comb n.5: ['lcavol', 'lweight', 'lbph', 'svi', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['age']
    RSS test: 14.300531209782973
    
    
    Comb n.6: ['lcavol', 'age', 'lbph', 'svi', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lweight']
    RSS test: 14.364169507348384
    
    
    Comb n.7: ['lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcavol']
    RSS test: 20.7394303261079
    
    
    At the end we have:
    min RSS: 12.690514046312394
    We drop: lbph
    ==========================================================================================
    
    ==========================================================================================
    Iter n.3:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'age', 'svi', 'lcp', 'gleason']
    Dropped features: [[], 'pgg45', 'lbph']
    
    Now we compare the model selecting 5 variables
    
    
    Comb n.1: ['lcavol', 'lweight', 'age', 'svi', 'lcp']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['gleason']
    RSS test: 12.70021855056121
    
    
    Comb n.2: ['lcavol', 'lweight', 'age', 'svi', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcp']
    RSS test: 11.497691985782552
    
    
    Comb n.3: ['lcavol', 'lweight', 'age', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['svi']
    RSS test: 14.720875322787679
    
    
    Comb n.4: ['lcavol', 'lweight', 'svi', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['age']
    RSS test: 13.183227370610318
    
    
    Comb n.5: ['lcavol', 'age', 'svi', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lweight']
    RSS test: 13.23436777636421
    
    
    Comb n.6: ['lweight', 'age', 'svi', 'lcp', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcavol']
    RSS test: 16.98150806110247
    
    
    At the end we have:
    min RSS: 11.497691985782552
    We drop: lcp
    ==========================================================================================
    
    ==========================================================================================
    Iter n.4:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'age', 'svi', 'gleason']
    Dropped features: [[], 'pgg45', 'lbph', 'lcp']
    
    Now we compare the model selecting 4 variables
    
    
    Comb n.1: ['lcavol', 'lweight', 'age', 'svi']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['gleason']
    RSS test: 11.6363032699816
    
    
    Comb n.2: ['lcavol', 'lweight', 'age', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['svi']
    RSS test: 14.030539349222018
    
    
    Comb n.3: ['lcavol', 'lweight', 'svi', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['age']
    RSS test: 11.956020656419863
    
    
    Comb n.4: ['lcavol', 'age', 'svi', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lweight']
    RSS test: 11.612572746859495
    
    
    Comb n.5: ['lweight', 'age', 'svi', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcavol']
    RSS test: 18.093182817451503
    
    
    At the end we have:
    min RSS: 11.612572746859495
    We drop: lweight
    ==========================================================================================
    
    ==========================================================================================
    Iter n.5:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'age', 'svi', 'gleason']
    Dropped features: [[], 'pgg45', 'lbph', 'lcp', 'lweight']
    
    Now we compare the model selecting 3 variables
    
    
    Comb n.1: ['lcavol', 'age', 'svi']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['gleason']
    RSS test: 11.71416952499475
    
    
    Comb n.2: ['lcavol', 'age', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['svi']
    RSS test: 14.18793001209106
    
    
    Comb n.3: ['lcavol', 'svi', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['age']
    RSS test: 11.484037587414818
    
    
    Comb n.4: ['age', 'svi', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcavol']
    RSS test: 19.78197534246865
    
    
    At the end we have:
    min RSS: 11.484037587414818
    We drop: age
    ==========================================================================================
    
    ==========================================================================================
    Iter n.6:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'svi', 'gleason']
    Dropped features: [[], 'pgg45', 'lbph', 'lcp', 'lweight', 'age']
    
    Now we compare the model selecting 2 variables
    
    
    Comb n.1: ['lcavol', 'svi']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['gleason']
    RSS test: 11.58358360007023
    
    
    Comb n.2: ['lcavol', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['svi']
    RSS test: 14.140851461764317
    
    
    Comb n.3: ['svi', 'gleason']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcavol']
    RSS test: 18.710131293690928
    
    
    At the end we have:
    min RSS: 11.58358360007023
    We drop: gleason
    ==========================================================================================
    
    ==========================================================================================
    Iter n.7:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'svi']
    Dropped features: [[], 'pgg45', 'lbph', 'lcp', 'lweight', 'age', 'gleason']
    
    Now we compare the model selecting 1 variables
    
    
    Comb n.1: ['lcavol']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['svi']
    RSS test: 14.392161587304827
    
    
    Comb n.2: ['svi']
    ------------------------------------------------------------------------------------------
    candidate dropped feature: ['lcavol']
    RSS test: 20.632078139876853
    
    
    At the end we have:
    min RSS: 14.392161587304827
    We drop: svi
    ==========================================================================================
    



```python
# look at the result
df_BackwardS
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>min_RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>R_squared</th>
      <th style='text-align:center; vertical-align:middle'>dropped_feature</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>[15.638220165228002]</td>
      <td style='text-align:center; vertical-align:middle'>15.638220</td>
      <td style='text-align:center; vertical-align:middle'>0.694371</td>
      <td style='text-align:center; vertical-align:middle'>[]</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, age, lbph, svi, lcp, gleason...</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>[13.492898446056923, 15.495404626758, 13.83423...</td>
      <td style='text-align:center; vertical-align:middle'>13.492898</td>
      <td style='text-align:center; vertical-align:middle'>0.678457</td>
      <td style='text-align:center; vertical-align:middle'>pgg45</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, age, lbph, svi, lcp, gleason]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>[13.43573406046562, 12.98960324749752, 15.2446...</td>
      <td style='text-align:center; vertical-align:middle'>12.690514</td>
      <td style='text-align:center; vertical-align:middle'>0.656353</td>
      <td style='text-align:center; vertical-align:middle'>lbph</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, age, svi, lcp, gleason]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>[12.70021855056121, 11.497691985782552, 14.720...</td>
      <td style='text-align:center; vertical-align:middle'>11.497692</td>
      <td style='text-align:center; vertical-align:middle'>0.645101</td>
      <td style='text-align:center; vertical-align:middle'>lcp</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, age, svi, gleason]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>[11.6363032699816, 14.030539349222018, 11.9560...</td>
      <td style='text-align:center; vertical-align:middle'>11.612573</td>
      <td style='text-align:center; vertical-align:middle'>0.561456</td>
      <td style='text-align:center; vertical-align:middle'>lweight</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, age, svi, gleason]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>5</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>[11.71416952499475, 14.18793001209106, 11.4840...</td>
      <td style='text-align:center; vertical-align:middle'>11.484038</td>
      <td style='text-align:center; vertical-align:middle'>0.561005</td>
      <td style='text-align:center; vertical-align:middle'>age</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi, gleason]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>6</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>[11.58358360007023, 14.140851461764317, 18.710...</td>
      <td style='text-align:center; vertical-align:middle'>11.583584</td>
      <td style='text-align:center; vertical-align:middle'>0.560532</td>
      <td style='text-align:center; vertical-align:middle'>gleason</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, svi]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>[14.392161587304827, 20.632078139876853]</td>
      <td style='text-align:center; vertical-align:middle'>14.392162</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>svi</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol]</td>
    </tr>
  </tbody>
</table>
</div>



### **Plot for Backward Selection**


```python
# Initialize the figure
width = 6
height = 6
nfig = 1
fig = plt.figure(figsize = (width*nfig,height))

# 1. RSS Test set plot
tmp_df = df_BackwardS;
ax = fig.add_subplot(1, nfig, 1)
for i in range(0,len(tmp_df.RSS_test)):
    ax.scatter([tmp_df.numb_features[i]]*(len(tmp_df.RSS_test[i])),tmp_df.RSS_test[i], alpha = .2, color = 'darkblue');
    
ax.set_xlabel('Subset Size k',fontsize=14);
ax.set_ylabel('RSS',fontsize=14);
ax.set_title('RSS on test set',fontsize=18);
ax.plot(tmp_df.numb_features,tmp_df.min_RSS_test,color = 'r', label = 'Best subset'); # line of best values
ax.grid(color='grey', linestyle='-', linewidth=0.5);
ax.legend();

fig.suptitle('Backward Subset Selection',fontsize=25, y=0.98);
fig.subplots_adjust(top=0.8)
plt.show()
```


![png](/posts/sl-ex3-subsetselection/output_44_0.png)


## **Comparison subset selection methods**
Comparison between Best, Forward and Backward Selction results on test set


```python
# Initialize the figure
width = 6
height = 6
nfig = 3
fig = plt.figure(figsize = (width*nfig,height))

# 1. BEST SUBSET SELECTION
tmp_df1 = df_BestS;           # scatter plot
tmp_df2 = df_BestS_RSS_test; # plot the line of best values
ax = fig.add_subplot(1, nfig, 1)
ax.scatter(tmp_df1.numb_features,tmp_df1.RSS_test, alpha = .2, color = 'darkblue');
ax.set_xlabel('Subset Size k',fontsize=14);
ax.set_ylabel('RSS',fontsize=14);
ax.set_title('Best Selection',fontsize=18);
ax.plot(tmp_df2.numb_features,tmp_df2.RSS_test,color = 'r', label = 'Best subset'); # line of best values
ax.grid(color='grey', linestyle='-', linewidth=0.5);
ax.legend();

# 2. FORWARD SUBSET SELECTION
tmp_df = df_ForwardS;
ax = fig.add_subplot(1, nfig, 2)
for i in range(0,len(tmp_df.RSS_test)):
    ax.scatter([tmp_df.numb_features[i]]*(len(tmp_df.RSS_test[i])),tmp_df.RSS_test[i], alpha = .2, color = 'darkblue');
    
ax.set_xlabel('Subset Size k',fontsize=14);
ax.set_ylabel('RSS',fontsize=14);
ax.set_title('Forward Selection',fontsize=18);
ax.plot(tmp_df.numb_features,tmp_df.min_RSS_test,color = 'r', label = 'Best subset'); # line of best values
ax.grid(color='grey', linestyle='-', linewidth=0.5);
ax.legend();

# 3. BACKWARD SUBSET SELECTION
tmp_df = df_BackwardS;
ax = fig.add_subplot(1, nfig, 3)
for i in range(0,len(tmp_df.RSS_test)):
    ax.scatter([tmp_df.numb_features[i]]*(len(tmp_df.RSS_test[i])),tmp_df.RSS_test[i], alpha = .2, color = 'darkblue');
    
ax.set_xlabel('Subset Size k',fontsize=14);
ax.set_ylabel('RSS',fontsize=14);
ax.set_title('Backward Selection',fontsize=18);
ax.plot(tmp_df.numb_features,tmp_df.min_RSS_test,color = 'r', label = 'Best subset'); # line of best values
ax.grid(color='grey', linestyle='-', linewidth=0.5);
ax.legend();

fig.suptitle('Comparison Subset Selection',fontsize=25, y=0.98);
fig.subplots_adjust(top=0.8)
plt.show()
```


![png](/posts/sl-ex3-subsetselection/output_46_0.png)



```python
# Initialize the figure
width = 6
height = 6
nfig = 1
fig = plt.figure(figsize = (width*nfig,height))
ax = fig.add_subplot(1, nfig, 1)

# Best Selection
ax.plot(df_BestS_RSS_test.numb_features,df_BestS_RSS_test.RSS_test,color = 'r', label = 'Best subset'); # line of best values

# Forward Selection
ax.plot(df_ForwardS.numb_features,df_ForwardS.min_RSS_test,color = 'b', label = 'Best subset'); # line of best values

# Backward Selection
ax.plot(df_BackwardS.numb_features,df_BackwardS.min_RSS_test,color = 'g', label = 'Best subset'); # line of best values

ax.grid(color='grey', linestyle='-', linewidth=0.5);
ax.set_xlabel('Subset Size k',fontsize=14);
ax.set_ylabel('RSS',fontsize=14);
ax.set_title('Comparison minimum RSS',fontsize=18);
ax.legend(['best','forward','backward'])
    
plt.show();
```


![png](/posts/sl-ex3-subsetselection/output_47_0.png)


<h4>Best method from Subset Selection</h4>

We see below the results of Best, Forward and Backward Subset Selection.


```python
df_range = [df_BestS_RSS_test, df_ForwardS, df_BackwardS]
columns_range = ['RSS_test','min_RSS_test','min_RSS_test']
methods = ['Best Selection', 'Forward Selection', 'Backward Selection']

for df, col, meth in zip(df_range,columns_range,methods):
    
    idx = df[col].idxmin()

    print(f"\nFor {meth} the best method has:\n\
n. features: {df['numb_features'][idx]}\n\
features: {df['features'][idx]}\n\
RSS test: {df[col][idx]}\n")
```

    
    For Best Selection the best method has:
    n. features: 3
    features: ('lcavol', 'svi', 'gleason')
    RSS test: 11.484037587414818
    
    
    For Forward Selection the best method has:
    n. features: 3
    features: ['lcavol', 'svi', 'gleason']
    RSS test: 11.484037587414818
    
    
    For Backward Selection the best method has:
    n. features: 3
    features: ['lcavol', 'svi', 'gleason']
    RSS test: 11.484037587414818
    
## **Backward selection with Z-score**

<h4>11. Perform backward selection using the z-score as a statistics for selecting the predictor to drop</h4>

    * Start from the full model
    * Remove at each step the variable having the smallest Z-score (which library is more suitable for this purpose?)


```python
# a flag to print some info, values {0,1}
flag = 1     # for short info
```


```python
## range
variables = data.columns.tolist()         # excluding 'const'
remaining_features = variables.copy()
tmpComb = remaining_features.copy()
dropped_features_list= []

## Initialize the list where we temporarily store data
RSS_test_list, R_squared_list = [], []
numb_features, features_list = [], []

# Loop over the number of variables
for k in range(len(variables),0,-1):

    # Compute the stats we need
    Zscores, minZ, idx_minZ = Zscore(X_train[tmpComb],Y_train)
    _, RSS_test, R_squared = LinReg(X_train[tmpComb], Y_train, X_test[tmpComb], Y_test)
  
    # Save stats
    RSS_test_list.append(RSS_test)
    R_squared_list.append(best_R_squared.copy())
    
    # Print some information, before upgrading the features
    if flag == 1:
        features = [remaining_features,dropped_features_list]
        params = [RSS_test,R_squared_list,tmpComb]
        if dropped_list: # only if dropped_feature is not an empty list
            dropped_feature = dropped_list[0]
            results = [idx_minZ,RSS_test]
        else:
            results = [RSS_test,[]]
            
        get_info_backwardS_Zscore(k,features,params,results,detailed)
        
    # Save features and number of features
    numb_features.append(k)
    features_list.append(tmpComb.copy())
    
    # update combinations
    tmpComb.remove(idx_minZ)
    dropped_list = list(set(remaining_features)-set(tmpComb))

    # Updating variables for next loop
    if dropped_list: # only if dropped_feature is not an empty list
        dropped_feature = dropped_list[0]
        remaining_features.remove(dropped_feature)
        dropped_features_list.append(dropped_feature)
    else:
        dropped_features_list.append([]) # at the initial iteration we drop nothing!
 
       
# Store in DataFrame
df_BackwardS_minZ = pd.DataFrame({'numb_features': numb_features,\
                             'RSS_test' : RSS_test_list,\
                             'R_squared': R_squared_list,\
                             'dropped_feature': dropped_features_list,\
                             'features': features_list})
```

    ==========================================================================================
    Iter n.0:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    Dropped features: []
    
    
    The Z-scores are:
     lcavol     5.366290
    lweight    2.750789
    age       -1.395909
    lbph       2.055846
    svi        2.469255
    lcp       -1.866913
    gleason   -0.146681
    pgg45      1.737840
    dtype: float64
    
    
    At the end we have:
    min RSS: 15.638220165228002
    We drop: gleason
    ==========================================================================================
    
    ==========================================================================================
    Iter n.1:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'pgg45']
    Dropped features: ['gleason']
    
    
    The Z-scores are:
     lcavol     5.462426
    lweight    2.833132
    age       -1.486490
    lbph       2.068796
    svi        2.519204
    lcp       -1.877253
    pgg45      2.182013
    dtype: float64
    
    
    At the end we have:
    min RSS: 15.495404626758
    We drop: age
    ==========================================================================================
    
    ==========================================================================================
    Iter n.2:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'lbph', 'svi', 'lcp', 'pgg45']
    Dropped features: ['gleason', 'age']
    
    
    The Z-scores are:
     lcavol     5.243670
    lweight    2.589758
    lbph       1.815545
    svi        2.544601
    lcp       -1.733570
    pgg45      1.871654
    dtype: float64
    
    
    At the end we have:
    min RSS: 16.457800398803407
    We drop: lcp
    ==========================================================================================
    
    ==========================================================================================
    Iter n.3:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'lbph', 'svi', 'pgg45']
    Dropped features: ['gleason', 'age', 'lcp']
    
    
    The Z-scores are:
     lcavol     4.899985
    lweight    2.551974
    lbph       1.952740
    svi        2.039752
    pgg45      1.190849
    dtype: float64
    
    
    At the end we have:
    min RSS: 14.577726321419373
    We drop: pgg45
    ==========================================================================================
    
    ==========================================================================================
    Iter n.4:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'lbph', 'svi']
    Dropped features: ['gleason', 'age', 'lcp', 'pgg45']
    
    
    The Z-scores are:
     lcavol     5.461353
    lweight    2.441316
    lbph       1.988469
    svi        2.458931
    dtype: float64
    
    
    At the end we have:
    min RSS: 13.689963661204882
    We drop: lbph
    ==========================================================================================
    
    ==========================================================================================
    Iter n.5:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight', 'svi']
    Dropped features: ['gleason', 'age', 'lcp', 'pgg45', 'lbph']
    
    
    The Z-scores are:
     lcavol     5.507417
    lweight    3.655671
    svi        1.985388
    dtype: float64
    
    
    At the end we have:
    min RSS: 12.015924403078804
    We drop: svi
    ==========================================================================================
    
    ==========================================================================================
    Iter n.6:
    
    At the beginning we have:
    Remaining features: ['lcavol', 'lweight']
    Dropped features: ['gleason', 'age', 'lcp', 'pgg45', 'lbph', 'svi']
    
    
    The Z-scores are:
     lcavol     7.938277
    lweight    3.582135
    dtype: float64
    
    
    At the end we have:
    min RSS: 14.77447043041511
    We drop: lweight
    ==========================================================================================
    
    ==========================================================================================
    Iter n.7:
    
    At the beginning we have:
    Remaining features: ['lcavol']
    Dropped features: ['gleason', 'age', 'lcp', 'pgg45', 'lbph', 'svi', 'lweight']
    
    
    The Z-scores are:
     lcavol    8.691694
    dtype: float64
    
    
    At the end we have:
    min RSS: 14.392161587304827
    We drop: lcavol
    ==========================================================================================
    


```python
df_BackwardS_minZ
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
      <th style='text-align:center; vertical-align:middle'>numb_features</th>
      <th style='text-align:center; vertical-align:middle'>RSS_test</th>
      <th style='text-align:center; vertical-align:middle'>R_squared</th>
      <th style='text-align:center; vertical-align:middle'>dropped_feature</th>
      <th style='text-align:center; vertical-align:middle'>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>15.638220</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>gleason</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, age, lbph, svi, lcp, gleason...</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>15.495405</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>age</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, age, lbph, svi, lcp, pgg45]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>16.457800</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>lcp</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, lbph, svi, lcp, pgg45]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>14.577726</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>pgg45</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, lbph, svi, pgg45]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>13.689964</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>lbph</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, lbph, svi]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>5</th>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>12.015924</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>svi</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight, svi]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>6</th>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>14.774470</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>lweight</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol, lweight]</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>14.392162</td>
      <td style='text-align:center; vertical-align:middle'>0.537516</td>
      <td style='text-align:center; vertical-align:middle'>lcavol</td>
      <td style='text-align:center; vertical-align:middle'>[lcavol]</td>
    </tr>
  </tbody>
</table>
</div>



<h4>12. Generate a chart having</h4>

        * x-axis: the subset size
        * y-axis: the RSS for the test set of the models generated at step 11
    * Compare it with the chart generated at point 10.


```python
# Initialize the figure
width = 6
height = 6
nfig = 1
fig = plt.figure(figsize = (width*nfig,height))

# 1. RSS Test set plot
tmp_df = df_BackwardS_minZ;
ax = fig.add_subplot(1, nfig, 1)
ax.scatter(tmp_df.numb_features,tmp_df.RSS_test, alpha = .2, color = 'darkblue');
    
ax.set_xlabel('Subset Size k',fontsize=14);
ax.set_ylabel('RSS',fontsize=14);
ax.set_title('RSS on test set',fontsize=18);
ax.plot(tmp_df.numb_features,tmp_df.RSS_test,color = 'r', label = 'Best subset'); # line of best values
ax.grid(color='grey', linestyle='-', linewidth=0.5);
ax.legend();

fig.suptitle('Backward Subset Selection',fontsize=25, y=0.98);
fig.subplots_adjust(top=0.85)
plt.show()
```


![png](/posts/sl-ex3-subsetselection/output_55_0.png)


### **Comparison with Z-score**

Now compare the results by choosing the best model not by minimizing RSS but maximizing Zscore.


```python
df_range = [df_BestS_RSS_test, df_ForwardS, df_BackwardS, df_BackwardS_minZ]
columns_range = ['RSS_test','min_RSS_test','min_RSS_test','RSS_test']
methods = ['Best Selection - RSS test', 'Forward Selection - RSS test', 'Backward Selection - RSS test', 'Backward Selection - Z score']

for df, col, meth in zip(df_range,columns_range,methods):
    
    idx = df[col].idxmin()

    print(f"\nFor {meth}, the best method has:\n\
n. features: {df['numb_features'][idx]}\n\
features: {df['features'][idx]}\n\
RSS test: {df[col][idx]}\n")
```

    
    For Best Selection - RSS test, the best method has:
    n. features: 3
    features: ('lcavol', 'svi', 'gleason')
    RSS test: 11.484037587414818
    
    
    For Forward Selection - RSS test, the best method has:
    n. features: 3
    features: ['lcavol', 'svi', 'gleason']
    RSS test: 11.484037587414818
    
    
    For Backward Selection - RSS test, the best method has:
    n. features: 3
    features: ['lcavol', 'svi', 'gleason']
    RSS test: 11.484037587414818
    
    
    For Backward Selection - Z score, the best method has:
    n. features: 3
    features: ['lcavol', 'lweight', 'svi']
    RSS test: 12.015924403078804
    

