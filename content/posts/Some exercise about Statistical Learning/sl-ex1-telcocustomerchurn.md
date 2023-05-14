+++
title = "SL1: Telco Customer Churn first data analysis using Python"
author = "Andrea Mortaro"	
layout = "notebook_toc"
showDate = false
weight = 6
draft = "false"
summary = " "
+++

{{< katex >}}


> <h4>Aim of the analysis</h4>The aim of this analysis is to predict behavior to retain customers.


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# data analysis and wrangling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv



```python
# Automatically Wrap Graph Labels in Matplotlib and Seaborn
# source: https://medium.com/dunder-data/automatically-wrap-graph-labels-in-matplotlib-and-seaborn-a48740bc9ce
import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
```

## **Analysis of the dataset Telcom_Customer_Churn**

<h4>1. Open the Telco Customer Churn dataset page in Kaggle.</h4>

> <h4>✅</h4>

<h4>2. Check the main properties of this dataset in the “Data” tab.</h4>

* How many samples (rows) does it have?
    > The number of samples (rows) is:  7043

* How many variables (columns)?
    > The number of variables (columns) is:  21

* What does each row/column represent?

    > Each row represents a customer, each column contains customer’s attributes.
    The data set includes information about:
    > * **Customers who left within the last month:**\
    the column is called "Churn"
    > * **Services that each customer has signed up for:**\
    phone, multiple lines, internet, online security, online backup,device protection, tech support, and streaming TV and movies
    > * **Customer account information:**\
    how long they’ve been a customer, contract, payment method,paperless billing, monthly charges, and total charges
    > * **Demographic info about customers:**\
    gender, age range, and if they have partners and dependents
    
* Which is the “target” column? What does it represent?
    > The target column is the "Churn"-column, because we want to predict this kind of behaviour.

<h4>3. Download the dataset into your computer.</h4>

* Which is the extension of the downloaded file?
    > **.zip**

<h4>4. Uncompress the file</h4>

* Which is the extension of the uncompressed file?
    > **.csv**

<h4>5. Open the uncompressed file by both a text editor and a spreadsheet software</h4>

- Which symbol is used to separate columns?
    > The comma!

- Which symbol is used to separate rows?
    > New line '\n'!

- Which values can you find for variable SeniorCitizen? And for variable Partner?
    > For `SeniorCitizen` the possible value is a boolean variable, where 1 means that the person is a senior citizen and 0 means no.\
    For `Partner` the possibile value is string variable, with `Yes' or 'No'.

<h4>6. Generate a new notebook for analyzing this dataset</h4>

* Hint: click on “New Kernel”, then choose the Notebook kernel type, on the right
* Assign the following title to the notebook:\
        "SL_L1_TelcoCustomerChurn_Surname"
* Then click on the “Commit” button on top-right to make the notebook ready to be started

> <h4>✅</h4>

<h4>7. Open the notebook documentation page to get help if needed</h4>

* Hint: click the “Docs” link on the right-bottom of your notebook page

> <h4>✅</h4>

<h4>8. Select the first cell (we will call it “Library import cell” in the following), run it</h4>

* What is the output of this action?
    > It loads some python packages
    
* What does the code “import numpy as np” do? Can you provide a reference website for this library?
    > Running `import numpy as np` I load numpy package, which is useful for linear algebra. You can find a reference [here](https://numpy.org/). 
    
* What does the code “import pandas as pd” do? Can you provide a reference website for this library?
    > Running `import pandas as pd` I load pandas package,m which is useful for data processing, and to handle CSV file I/O (e.g. pd.read_csv). You can find a reference [here](https://pandas.pydata.org/).
    
* What does the code “import os” do? Can you provide a reference website for this library?
    > Running `import os` I load os package, which is useful for operating system functionality. There is a reference [here](https://docs.python.org/3/library/os.html).
    
* How many data files are available? Please provide their names.
    > With os module I can list all the files available, and in our case there's only the file named "/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv".

<h4>9. Add to the first cell new lines to load the following libraries: seaborn, matplotlib.pyplot, sklearn.linear_model (only LogisticRegression)</h4>

* Hint: find similar code in the Titanic notebook if needed

> <h4>✅</h4>

<h4>10. Select the first cell and add a new cell on top of it</h4>

* Hint: use the button on top-right of the cell

> <h4>✅</h4>

<h4>11. Select the new cell and transform it in a “Markdown” cell, then copy all the text in this pdf file and paste it in the new Markdown cell</h4>

> <h4>✅</h4>

<h4>12. Please write your answers to the questions above in the new Markdown cell. From now on you can use the same cell to write your answers as well</h4>

> <h4>✅</h4>

<h4>13. Select the “Library input cell” and add a new cell below it</h4>

> <h4>✅</h4>

<h4>14. Use the new cell to load the Telco Customer Churn dataset into a Pandas DataFrame variable called data</h4>

* Hint: find similar code in the Titanic notebook if needed
* Remind to run the cell after writing the code-box

> <h4>✅</h4>

<h4>15. Add the following `comment` before data loading line: “Data acquisition”</h4>

> <h4>✅</h4>

<h4>16. Add also a `Markdown cell` before the data loading cell and write in bold the text “Data acquisition”</h4>

* Markdown cells should be used to give a structure to the report, hence they should be added before each new section

> <h4>✅</h4>

## **Data acquisition**


```python
## Data acquisition
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
```

<h4>17. In a new cell show the number of rows, the number of columns, and the total number of cells in the dataset</h4>

* Hint: display the related parameters of the `Pandas DataFrame`
* Hint: use the print function to print the results
* You should print, in particular, the following strings:
    * “The number of customers is XXXX”
    * “The number of variables is YYYY”
    * “The total number of cells is ZZZZ”
* Other hints:
    * How can you select a single element from the shape tuple?
    * How can you convert a number to string?
    * How can you concatenate two strings?
    * How can you print the final string?

## **Data analysis**


```python
#Dataset dimension

[num_rows,num_columns]=data.shape;
num_elements=data.size;
print(f"The number of costumers (rows) is {num_rows}.\n\
The number of variables (columns) is {num_columns}.\n\
The number of elements is {num_elements}.\n")
print("To select a single element from the shape tuple it's enough to assing the return of `data.shape` to two variables.\n\
So I get a single element by calling one of them.\n\n\
I can convert a number to string by call the method `str`.\n\
I can concatenate two strings by using `+` between the two strings and I print the final string by calling `print` with the argument `\"string1\"+\"string2\"")
```

    The number of costumers (rows) is 7043.
    The number of variables (columns) is 21.
    The number of elements is 147903.
    
    To select a single element from the shape tuple it's enough to assing the return of `data.shape` to two variables.
    So I get a single element by calling one of them.
    
    I can convert a number to string by call the method `str`.
    I can concatenate two strings by using `+` between the two strings and I print the final string by calling `print` with the argument `"string1"+"string2"


<h4>18. Add the following `comment` at the beginning of the cell: “Dataset dimension”</h4>

> <h4>✅</h4>

<h4>19. Add a new `markdown cell` before this cell and write in it the title “Data Analysis”</h4>

> <h4>✅</h4>

<h4>20. In a new cell show the names of the variables in the dataset</h4>

* Hint: print the column’s names of variable data
> <h4>✅</h4>


```python
print("The names of the variables in the dataset are:")

for num, column in zip(range(1,len(data.columns)+1),data.columns):
    print(f"{num}. {column}")
```

    The names of the variables in the dataset are:
    1. customerID
    2. gender
    3. SeniorCitizen
    4. Partner
    5. Dependents
    6. tenure
    7. PhoneService
    8. MultipleLines
    9. InternetService
    10. OnlineSecurity
    11. OnlineBackup
    12. DeviceProtection
    13. TechSupport
    14. StreamingTV
    15. StreamingMovies
    16. Contract
    17. PaperlessBilling
    18. PaymentMethod
    19. MonthlyCharges
    20. TotalCharges
    21. Churn


<h4>21. In a new cell show the first and last 10 rows in the dataset</h4>

* Hint: find the correct DataFrame methods in the Pandas’ documentation


```python
print("The first 10 rows in the dataset are:")
data.head(10)
```

    The first 10 rows in the dataset are:



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
      <th style='text-align:center; vertical-align:middle'>customerID</th>
      <th style='text-align:center; vertical-align:middle'>gender</th>
      <th style='text-align:center; vertical-align:middle'>SeniorCitizen</th>
      <th style='text-align:center; vertical-align:middle'>Partner</th>
      <th style='text-align:center; vertical-align:middle'>Dependents</th>
      <th style='text-align:center; vertical-align:middle'>tenure</th>
      <th style='text-align:center; vertical-align:middle'>PhoneService</th>
      <th style='text-align:center; vertical-align:middle'>MultipleLines</th>
      <th style='text-align:center; vertical-align:middle'>InternetService</th>
      <th style='text-align:center; vertical-align:middle'>OnlineSecurity</th>
      <th style='text-align:center; vertical-align:middle'>...</th>
      <th style='text-align:center; vertical-align:middle'>DeviceProtection</th>
      <th style='text-align:center; vertical-align:middle'>TechSupport</th>
      <th style='text-align:center; vertical-align:middle'>StreamingTV</th>
      <th style='text-align:center; vertical-align:middle'>StreamingMovies</th>
      <th style='text-align:center; vertical-align:middle'>Contract</th>
      <th style='text-align:center; vertical-align:middle'>PaperlessBilling</th>
      <th style='text-align:center; vertical-align:middle'>PaymentMethod</th>
      <th style='text-align:center; vertical-align:middle'>MonthlyCharges</th>
      <th style='text-align:center; vertical-align:middle'>TotalCharges</th>
      <th style='text-align:center; vertical-align:middle'>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>7590-VHVEG</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No phone service</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Electronic check</td>
      <td style='text-align:center; vertical-align:middle'>29.85</td>
      <td style='text-align:center; vertical-align:middle'>29.85</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>5575-GNVDE</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>34</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>One year</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Mailed check</td>
      <td style='text-align:center; vertical-align:middle'>56.95</td>
      <td style='text-align:center; vertical-align:middle'>1889.5</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>3668-QPYBK</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Mailed check</td>
      <td style='text-align:center; vertical-align:middle'>53.85</td>
      <td style='text-align:center; vertical-align:middle'>108.15</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>7795-CFOCW</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>45</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No phone service</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>One year</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Bank transfer (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>42.30</td>
      <td style='text-align:center; vertical-align:middle'>1840.75</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>9237-HQITU</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Electronic check</td>
      <td style='text-align:center; vertical-align:middle'>70.70</td>
      <td style='text-align:center; vertical-align:middle'>151.65</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>5</th>
      <td style='text-align:center; vertical-align:middle'>9305-CDSKC</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Electronic check</td>
      <td style='text-align:center; vertical-align:middle'>99.65</td>
      <td style='text-align:center; vertical-align:middle'>820.5</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>6</th>
      <td style='text-align:center; vertical-align:middle'>1452-KIOVK</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>22</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Credit card (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>89.10</td>
      <td style='text-align:center; vertical-align:middle'>1949.4</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7</th>
      <td style='text-align:center; vertical-align:middle'>6713-OKOMC</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>10</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No phone service</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Mailed check</td>
      <td style='text-align:center; vertical-align:middle'>29.75</td>
      <td style='text-align:center; vertical-align:middle'>301.9</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>8</th>
      <td style='text-align:center; vertical-align:middle'>7892-POOKP</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>28</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Electronic check</td>
      <td style='text-align:center; vertical-align:middle'>104.80</td>
      <td style='text-align:center; vertical-align:middle'>3046.05</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>9</th>
      <td style='text-align:center; vertical-align:middle'>6388-TABGU</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>62</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>One year</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Bank transfer (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>56.15</td>
      <td style='text-align:center; vertical-align:middle'>3487.95</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 21 columns</p>
</div>




```python
print("The last 10 rows in the dataset are:")
data.tail(10)
```

    The last 10 rows in the dataset are:




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
    <tr style="text-align: left;">
      <th style='text-align:center; vertical-align:middle'></th>
      <th style='text-align:center; vertical-align:middle'>customerID</th>
      <th style='text-align:center; vertical-align:middle'>gender</th>
      <th style='text-align:center; vertical-align:middle'>SeniorCitizen</th>
      <th style='text-align:center; vertical-align:middle'>Partner</th>
      <th style='text-align:center; vertical-align:middle'>Dependents</th>
      <th style='text-align:center; vertical-align:middle'>tenure</th>
      <th style='text-align:center; vertical-align:middle'>PhoneService</th>
      <th style='text-align:center; vertical-align:middle'>MultipleLines</th>
      <th style='text-align:center; vertical-align:middle'>InternetService</th>
      <th style='text-align:center; vertical-align:middle'>OnlineSecurity</th>
      <th style='text-align:center; vertical-align:middle'>...</th>
      <th style='text-align:center; vertical-align:middle'>DeviceProtection</th>
      <th style='text-align:center; vertical-align:middle'>TechSupport</th>
      <th style='text-align:center; vertical-align:middle'>StreamingTV</th>
      <th style='text-align:center; vertical-align:middle'>StreamingMovies</th>
      <th style='text-align:center; vertical-align:middle'>Contract</th>
      <th style='text-align:center; vertical-align:middle'>PaperlessBilling</th>
      <th style='text-align:center; vertical-align:middle'>PaymentMethod</th>
      <th style='text-align:center; vertical-align:middle'>MonthlyCharges</th>
      <th style='text-align:center; vertical-align:middle'>TotalCharges</th>
      <th style='text-align:center; vertical-align:middle'>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7033</th>
      <td style='text-align:center; vertical-align:middle'>9767-FFLEM</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>38</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Credit card (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>69.50</td>
      <td style='text-align:center; vertical-align:middle'>2625.25</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7034</th>
      <td style='text-align:center; vertical-align:middle'>0639-TSIQW</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>67</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Credit card (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>102.95</td>
      <td style='text-align:center; vertical-align:middle'>6886.25</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7035</th>
      <td style='text-align:center; vertical-align:middle'>8456-QDAVC</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>19</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Bank transfer (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>78.70</td>
      <td style='text-align:center; vertical-align:middle'>1495.1</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7036</th>
      <td style='text-align:center; vertical-align:middle'>7750-EYXWZ</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>12</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No phone service</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>One year</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Electronic check</td>
      <td style='text-align:center; vertical-align:middle'>60.65</td>
      <td style='text-align:center; vertical-align:middle'>743.3</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7037</th>
      <td style='text-align:center; vertical-align:middle'>2569-WGERO</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>72</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No internet service</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No internet service</td>
      <td style='text-align:center; vertical-align:middle'>No internet service</td>
      <td style='text-align:center; vertical-align:middle'>No internet service</td>
      <td style='text-align:center; vertical-align:middle'>No internet service</td>
      <td style='text-align:center; vertical-align:middle'>Two year</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Bank transfer (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>21.15</td>
      <td style='text-align:center; vertical-align:middle'>1419.4</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7038</th>
      <td style='text-align:center; vertical-align:middle'>6840-RESVB</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>24</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>One year</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Mailed check</td>
      <td style='text-align:center; vertical-align:middle'>84.80</td>
      <td style='text-align:center; vertical-align:middle'>1990.5</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7039</th>
      <td style='text-align:center; vertical-align:middle'>2234-XADUH</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>72</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>One year</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Credit card (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>103.20</td>
      <td style='text-align:center; vertical-align:middle'>7362.9</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7040</th>
      <td style='text-align:center; vertical-align:middle'>4801-JZAZL</td>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>11</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No phone service</td>
      <td style='text-align:center; vertical-align:middle'>DSL</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Electronic check</td>
      <td style='text-align:center; vertical-align:middle'>29.60</td>
      <td style='text-align:center; vertical-align:middle'>346.45</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7041</th>
      <td style='text-align:center; vertical-align:middle'>8361-LTMKD</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Mailed check</td>
      <td style='text-align:center; vertical-align:middle'>74.40</td>
      <td style='text-align:center; vertical-align:middle'>306.6</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>7042</th>
      <td style='text-align:center; vertical-align:middle'>3186-AJIEK</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>66</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Two year</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Bank transfer (automatic)</td>
      <td style='text-align:center; vertical-align:middle'>105.65</td>
      <td style='text-align:center; vertical-align:middle'>6844.5</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 21 columns</p>
</div>



<h4>22. In a new cell show i) the type of variable data, ii) the number of missing values for each variable, iii) the type of each variable, iv) the total memory used to store variable data</h4>

* Hint: all this information can be provided by a single method of DataFrame
With the method `.info` we can observe all the info asked.


```python
data.info(verbose=True,memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   customerID        7043 non-null   object 
     1   gender            7043 non-null   object 
     2   SeniorCitizen     7043 non-null   int64  
     3   Partner           7043 non-null   object 
     4   Dependents        7043 non-null   object 
     5   tenure            7043 non-null   int64  
     6   PhoneService      7043 non-null   object 
     7   MultipleLines     7043 non-null   object 
     8   InternetService   7043 non-null   object 
     9   OnlineSecurity    7043 non-null   object 
     10  OnlineBackup      7043 non-null   object 
     11  DeviceProtection  7043 non-null   object 
     12  TechSupport       7043 non-null   object 
     13  StreamingTV       7043 non-null   object 
     14  StreamingMovies   7043 non-null   object 
     15  Contract          7043 non-null   object 
     16  PaperlessBilling  7043 non-null   object 
     17  PaymentMethod     7043 non-null   object 
     18  MonthlyCharges    7043 non-null   float64
     19  TotalCharges      7043 non-null   object 
     20  Churn             7043 non-null   object 
    dtypes: float64(1), int64(2), object(18)
    memory usage: 7.8 MB



```python
print("\n'TotalCharges' contains also empty value and it will be difficult to handle,\
for instance data.iloc[488]['TotalCharges'] contains: ",data.iloc[488]['TotalCharges'],".\n\
A possible solution is to cast these two variables.")
```
    
    'TotalCharges' contains also empty value and it will be difficult to handle,for instance data.iloc[488]['TotalCharges'] contains:    .
    A possible solution is to cast these two variables.

{{< alert >}}
**WARNING**: Pay attention to 'customerID' and 'TotalCharges' variables because they are object, but they contains numerical values!
{{< /alert >}}

* How many missing values are there in total?
    > To detect missing values I use the method `.isnull()`:


```python
# # Count total NaN at each column in a DataFrame
# print(" \nCount total NaN at each column in a DataFrame:\n",\
#       data.isnull().sum(),"\n")

# # Count total NaN at each row in a DataFrame
# for i in range(len(data.index)) :
#     print(" Total NaN in row", i + 1, ":",
#           data.iloc[i].isnull().sum())

print("There are ",data.isnull().sum().sum()," missing values in total.")
```

    There are  0  missing values in total.


* Which variables are categorical?


```python
print("\nThe categorical variables are:")

objVar = data.select_dtypes(include=['object']).columns.tolist()

for num, col in zip(range(1,len(objVar)+1),objVar):
    print(f"{num}. {col}")
```

    
    The categorical variables are:
    1. customerID
    2. gender
    3. Partner
    4. Dependents
    5. PhoneService
    6. MultipleLines
    7. InternetService
    8. OnlineSecurity
    9. OnlineBackup
    10. DeviceProtection
    11. TechSupport
    12. StreamingTV
    13. StreamingMovies
    14. Contract
    15. PaperlessBilling
    16. PaymentMethod
    17. TotalCharges
    18. Churn


* Which variables are numerical?


```python
print("\nThe numerical variables are:")

    
numVar = data.select_dtypes(exclude=['object']).columns.tolist()

for num, col in zip(range(1,len(objVar)+1),numVar):
    print(f"{num}. {col}")
```

    
    The numerical variables are:
    1. SeniorCitizen
    2. tenure
    3. MonthlyCharges


<h4>23. In a new cell show the following basic statistics for all `numerical variables`: number of non-missing values, mean, standard deviation, minimum, maximum, median, 1 st and 3 rd quartiles</h4>
* Hint: all this information can be provided by a single method of DataFrame


```python
data.describe(percentiles=[.25, .5, .75])
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
      <th style='text-align:center; vertical-align:middle'>SeniorCitizen</th>
      <th style='text-align:center; vertical-align:middle'>tenure</th>
      <th style='text-align:center; vertical-align:middle'>MonthlyCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>count</th>
      <td style='text-align:center; vertical-align:middle'>7043.000000</td>
      <td style='text-align:center; vertical-align:middle'>7043.000000</td>
      <td style='text-align:center; vertical-align:middle'>7043.000000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>mean</th>
      <td style='text-align:center; vertical-align:middle'>0.162147</td>
      <td style='text-align:center; vertical-align:middle'>32.371149</td>
      <td style='text-align:center; vertical-align:middle'>64.761692</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>std</th>
      <td style='text-align:center; vertical-align:middle'>0.368612</td>
      <td style='text-align:center; vertical-align:middle'>24.559481</td>
      <td style='text-align:center; vertical-align:middle'>30.090047</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>min</th>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>18.250000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>25%</th>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>9.000000</td>
      <td style='text-align:center; vertical-align:middle'>35.500000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>50%</th>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>29.000000</td>
      <td style='text-align:center; vertical-align:middle'>70.350000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>75%</th>
      <td style='text-align:center; vertical-align:middle'>0.000000</td>
      <td style='text-align:center; vertical-align:middle'>55.000000</td>
      <td style='text-align:center; vertical-align:middle'>89.850000</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>max</th>
      <td style='text-align:center; vertical-align:middle'>1.000000</td>
      <td style='text-align:center; vertical-align:middle'>72.000000</td>
      <td style='text-align:center; vertical-align:middle'>118.750000</td>
    </tr>
  </tbody>
</table>
</div>



<h4>24. In a new cell show the following basic information for all `categorical variables`: number of non-missing values, number of unique values, most frequent value and frequency of the most frequent value.</h4>
* Hint: all this information can be provided by the DataFrame method used in question 22, using specific arguments
* Can you see any strange value in this result?


```python
data.describe(include=['object'])
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
      <th style='text-align:center; vertical-align:middle'>customerID</th>
      <th style='text-align:center; vertical-align:middle'>gender</th>
      <th style='text-align:center; vertical-align:middle'>Partner</th>
      <th style='text-align:center; vertical-align:middle'>Dependents</th>
      <th style='text-align:center; vertical-align:middle'>PhoneService</th>
      <th style='text-align:center; vertical-align:middle'>MultipleLines</th>
      <th style='text-align:center; vertical-align:middle'>InternetService</th>
      <th style='text-align:center; vertical-align:middle'>OnlineSecurity</th>
      <th style='text-align:center; vertical-align:middle'>OnlineBackup</th>
      <th style='text-align:center; vertical-align:middle'>DeviceProtection</th>
      <th style='text-align:center; vertical-align:middle'>TechSupport</th>
      <th style='text-align:center; vertical-align:middle'>StreamingTV</th>
      <th style='text-align:center; vertical-align:middle'>StreamingMovies</th>
      <th style='text-align:center; vertical-align:middle'>Contract</th>
      <th style='text-align:center; vertical-align:middle'>PaperlessBilling</th>
      <th style='text-align:center; vertical-align:middle'>PaymentMethod</th>
      <th style='text-align:center; vertical-align:middle'>TotalCharges</th>
      <th style='text-align:center; vertical-align:middle'>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>count</th>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>7043</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>unique</th>
      <td style='text-align:center; vertical-align:middle'>7043</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>6531</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>top</th>
      <td style='text-align:center; vertical-align:middle'>7590-VHVEG</td>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Fiber optic</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Month-to-month</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>Electronic check</td>
      <td style='text-align:center; vertical-align:middle'></td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>freq</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>3555</td>
      <td style='text-align:center; vertical-align:middle'>3641</td>
      <td style='text-align:center; vertical-align:middle'>4933</td>
      <td style='text-align:center; vertical-align:middle'>6361</td>
      <td style='text-align:center; vertical-align:middle'>3390</td>
      <td style='text-align:center; vertical-align:middle'>3096</td>
      <td style='text-align:center; vertical-align:middle'>3498</td>
      <td style='text-align:center; vertical-align:middle'>3088</td>
      <td style='text-align:center; vertical-align:middle'>3095</td>
      <td style='text-align:center; vertical-align:middle'>3473</td>
      <td style='text-align:center; vertical-align:middle'>2810</td>
      <td style='text-align:center; vertical-align:middle'>2785</td>
      <td style='text-align:center; vertical-align:middle'>3875</td>
      <td style='text-align:center; vertical-align:middle'>4171</td>
      <td style='text-align:center; vertical-align:middle'>2365</td>
      <td style='text-align:center; vertical-align:middle'>11</td>
      <td style='text-align:center; vertical-align:middle'>5174</td>
    </tr>
  </tbody>
</table>
</div>



## **Visualization of data**

<h4>25. In a new cell show the histograms of each numeric variable (i.e., column) in the dataset</h4>

* Hint: try to find a specific method in the DataFrame API documentation


```python
# numdata = data.select_dtypes(include=['number'])
# cols = numdata.columns.values
numdata = data._get_numeric_data()
# numdata.hist(bins=20)

# fig properties
row = 1
col = 3
width = 20
height = 4

# initialize the figure
fig, axes = plt.subplots(row, col,figsize=(width,height))

for ax,numcol in zip(axes.flatten(),numdata.columns.tolist()):
    numdata.hist(column=numcol,ax=ax)
    
plt.show(fig) # force to show the plot after the print
```


![png](/posts/sl-l1-telcocustomerchurn/output_34_0.png)


<h4>26. In a new cell show the box-plots of each numeric variable (i.e., column) in the dataset</h4>

* Hint: try to find a specific method in the DataFrame API documentation
* Does this chart provide a good visualization? Why?
* Try to generate one box-plot for each numerical variable
* Try to put all three charts in the same figure using the subplot function


```python
print("Not a good visualization because it groups together all the numerical variables:\n")
fig1 = numdata.boxplot()
plt.show(fig1) # force to show the plot after the print

print("\nMoreover for 'SeniorCitizen' variable we can not distinguish all the info from the plot.")
```

    Not a good visualization because it groups together all the numerical variables:
    



![png](/posts/sl-l1-telcocustomerchurn/output_36_1.png)


    
    Moreover for 'SeniorCitizen' variable we can not distinguish all the info from the plot.



```python
print("To have a good visualization, we can plot each numerical variable in a single boxplot as follows:")

# fig properties
row = 1
col = 3
width = 20
height = 4

# initialize the figure
fig2, axes = plt.subplots(row, col,figsize=(width,height))
# fig.tight_layout()
# #fig.subplots_adjust(wspace=0.2)

for ax,numcol in zip(np.ravel(axes),numdata.columns.tolist()):
    numdata.boxplot(column=numcol,ax=ax)

plt.show(fig2) # force to show the plot after the print
```

    To have a good visualization, we can plot each numerical variable in a single boxplot as follows:



![png](/posts/sl-l1-telcocustomerchurn/output_37_1.png)


<h4>27. In a new cell show the histograms of the categorical variables in the dataset</h4>

* Hint: try to use a function from the Seaborn library which counts the number of time each element appears and makes a related bar plot
* Hint: use the subplot function to put all the charts in the same figure
* Hint: resize the figure so that to avoid overlapping and enable a clear visualization of all charts


```python
catdata = data.select_dtypes(include=["object"]);

# fig properties
row = 4
col = 4
width = 18
height = 10

fig, axes = plt.subplots(row, col,figsize=(width,height))

# Dropping 'costumerID' and 'TotalCharges' variables because makes too hard to plot!
categorical_variables = catdata[catdata.columns.difference(['customerID','TotalCharges'])].columns.tolist()

for ax,col in zip(axes.flatten(),categorical_variables):
    sns.countplot(data=data, x=col,ax=ax)
    
    # Since for 'PaymentMethod' we have xticklabels which are overlapping    
    if col == 'PaymentMethod':
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right") # rotate x-tick-labels
        wrap_labels(ax, 10) # wrap the words at or before the 10th character

fig.tight_layout()

plt.show(fig) # force to show the plot after the print
```


![png](/posts/sl-l1-telcocustomerchurn/output_39_0.png)


<h4>28. In a new cell generate a new DataFrame called data1 and containing only variables gender, Partner, MonthlyCharges, Churn</h4>
* Hint: you could try also other selections


```python
#data1 = data[['gender','Partner','MonthlyCharges','Churn']]
data1 = pd.DataFrame(data,columns=['gender','Partner','MonthlyCharges','Churn'])
```

<h4>29. In a new cell show the first 5 rows of the new dataset</h4>


```python
data1.head(5)
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
      <th style='text-align:center; vertical-align:middle'>gender</th>
      <th style='text-align:center; vertical-align:middle'>Partner</th>
      <th style='text-align:center; vertical-align:middle'>MonthlyCharges</th>
      <th style='text-align:center; vertical-align:middle'>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>29.85</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>56.95</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>53.85</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>42.30</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>70.70</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
  </tbody>
</table>
</div>



## **Prepare the data**

<h4>30. Convert categorical values in data1 to numeric as follows:</h4>

* **gender: Male=0, Female=1**
* **Partner: No=0, Yes=1**
* **Churn: No=0, Yes=1**

* Hint: find similar code in the Titanic notebook if needed


```python
gender_mapping = {'Male': 0, 'Female': 1}
mapping = {'No': 0, 'Yes': 1}
data1['gender'] = data1['gender'].map(gender_mapping)
# data1['gender'] = data1['gender'].astype(int) # it doesn't need to cast
data1['Partner'] = data1['Partner'].map(mapping)
data1['Churn'] = data1['Churn'].map(mapping)

display(data.loc[:,['gender', 'Partner','Churn']].head())
display(data1.loc[:,['gender', 'Partner','Churn']].head())
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
      <th style='text-align:center; vertical-align:middle'>gender</th>
      <th style='text-align:center; vertical-align:middle'>Partner</th>
      <th style='text-align:center; vertical-align:middle'>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>Male</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>Female</td>
      <td style='text-align:center; vertical-align:middle'>No</td>
      <td style='text-align:center; vertical-align:middle'>Yes</td>
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
    <tr style="text-align: right;">
      <th style='text-align:center; vertical-align:middle'></th>
      <th style='text-align:center; vertical-align:middle'>gender</th>
      <th style='text-align:center; vertical-align:middle'>Partner</th>
      <th style='text-align:center; vertical-align:middle'>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>1</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>1</td>
    </tr>
  </tbody>
</table>
</div>


<h4>31. Generate a separate Series variable called data1Churn for the dependent (churn) variable and drop it from DataFrame data1</h4>

* Hint: Series is a data structure defined in Pandas, try to find its documentation page
* Hint: each column of a DataFrame is a Series
* Hint: learn how to drop columns from a dataset in the Titanoc notebook
* What is the difference between data1[[‘Churn’]] and data1[‘Churn’]?
* When single square brackets are used with Pandas DataFrame? When double brackets are used instead?


```python
## Genereate a series variable:
data1Churn = data1['Churn']
## Dropping "Churn" variable from data1
# data1 = data1.drop("Churn",axis=1) # don't run this twice!
data1 = data1[['gender','Partner','MonthlyCharges']]
```

The difference between `data1[[‘Churn’]]` and `data1[‘Churn’]` is that:
> `data1[[‘Churn’]]` return a Pandas DataFrame, while `data1[‘Churn’]` returns a Pandas Series.

So according to our needs we choose the more appropriate class to use.

## **Linear Logistic Regression**

<h4>32. Generate a linear logistic model using data1 as independent variables and data1Churn as dependent variable, then show the model “score”</h4>

* Hint: try to find a function for linear logistic model learning in the sklearn library
* Hint: find similar code in the Titanic notebook if needed


```python
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(data1, data1Churn)
Y_pred = logreg.predict(data1)# test on data1
acc_log = round(logreg.score(data1,data1Churn), 2) # R2
print("The score of the Logistic Regression model is: ",acc_log)
```

    The score of the Logistic Regression model is:  0.72


<h4>33. Show the parameters of the linear logistic model computed above. Which variable seems to be more related to customer churn?</h4>

* Hint: find similar code in the Titanic notebook if needed


```python
logreg.get_params(deep=True)
# penalty: Specify the norm of the penalty
# for other info see here:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.get_params

print("The intercept is: ",logreg.intercept_[0],"\n\
The coefficients of our model are: ",logreg.coef_[0][:])
```

    The intercept is:  -1.8510896831170833 
    The coefficients of our model are:  [ 0.01730586 -0.83873431  0.01754114]



```python
# Correlation
coeff_data1 = pd.DataFrame(data1.columns)
coeff_data1.columns = ['Feature']
coeff_data1["Correlation"] = pd.Series(logreg.coef_[0])

coeff_data1.sort_values(by='Correlation', ascending=False)
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
      <th style='text-align:center; vertical-align:middle'>Feature</th>
      <th style='text-align:center; vertical-align:middle'>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>MonthlyCharges</td>
      <td style='text-align:center; vertical-align:middle'>0.017541</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>gender</td>
      <td style='text-align:center; vertical-align:middle'>0.017306</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>Partner</td>
      <td style='text-align:center; vertical-align:middle'>-0.838734</td>
    </tr>
  </tbody>
</table>
</div>

