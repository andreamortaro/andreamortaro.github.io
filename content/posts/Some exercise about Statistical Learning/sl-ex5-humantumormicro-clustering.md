+++
title = "SL5: Human Tumor Microarray dataset - clustering with k-means"
author = "Andrea Mortaro"
layout = "single"
showDate = false
weight = 2
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
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from IPython.display import Image # to visualize images
from tabulate import tabulate # to create tables

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

    /kaggle/input/14cancer/chart.png
    /kaggle/input/14cancer/14cancer.ytrain.txt
    /kaggle/input/14cancer/14cancer.xtrain.txt
    /kaggle/input/14cancer/chart2.png


<h4>1. Download the “14-cancer microarray data” from the <a rel="nofollow" href="https://web.stanford.edu/~hastie/ElemStatLearn/" hreflang="eng" target="_blank">book website</a></h4>

* Get information about the dataset in file 14cancer.info and in Chapter 1 (page 5) of the book (Hastie et al., 2009)

Some info about <kbd>14cancer.xtrain.txt</kbd> and<kbd>14cancer.ytrain.txt</kbd>.

* DNA microarrays measure the expression of genes in a cell

* 14-cancer gene expression data set:
    * 16064 genes
    * 144 training samples
    * 54 test samples

* One gene per row, one sample per column.

* Cancer classes are labelled as follows:
    1.  breast
    2.  prostate
    3.  lung
    4.  collerectal
    5.  lymphoma
    6.  bladder
    7.  melanoma
    8.  uterus
    9.  leukemia
    10. renal
    11. pancreas
    12. ovary
    13. meso
    14. cns

<h4>2. Generate a new Kernel and give it the name:</h4>
        “SL_EX5_HTM_Clustering_Surname”

<h4>3. Load the data in Kaggle</h4>

## **Data acquisition**


```python
# Load the Cancer Microarray dataset (already splitted in train and test)
xtrain = pd.read_csv('/kaggle/input/14cancer/14cancer.xtrain.txt', sep='\s+',header=None)
ytrain = pd.read_csv('/kaggle/input/14cancer/14cancer.ytrain.txt',sep='\s+',header=None)
```

{{< alert >}}
<strong>Warning:</strong> the dataset is already splitted in training set and test set.
{{< /alert >}}



## **Data pre-processing**


```python
xtrain = xtrain.transpose() # The columns represent the genes, and the rows are the different samples
ytrain = ytrain.transpose() # for each sample I have a label

(n_samples, n_genes), n_labels = xtrain.shape, np.unique(ytrain).size
print(f"#genes: {n_genes}, #samples: {n_samples}, #labels {n_labels}")
```

    #genes: 16063, #samples: 144, #labels 14



{{< alert >}}
<strong>Warning:</strong> I don't standardize the data before to perform clustering, in order to do not loose the natural properties of my dataset.
{{< /alert >}}


```python
xtrain
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
      <th style='text-align:center; vertical-align:middle'>0</th>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <th style='text-align:center; vertical-align:middle'>5</th>
      <th style='text-align:center; vertical-align:middle'>6</th>
      <th style='text-align:center; vertical-align:middle'>7</th>
      <th style='text-align:center; vertical-align:middle'>8</th>
      <th style='text-align:center; vertical-align:middle'>9</th>
      <th style='text-align:center; vertical-align:middle'>...</th>
      <th style='text-align:center; vertical-align:middle'>16053</th>
      <th style='text-align:center; vertical-align:middle'>16054</th>
      <th style='text-align:center; vertical-align:middle'>16055</th>
      <th style='text-align:center; vertical-align:middle'>16056</th>
      <th style='text-align:center; vertical-align:middle'>16057</th>
      <th style='text-align:center; vertical-align:middle'>16058</th>
      <th style='text-align:center; vertical-align:middle'>16059</th>
      <th style='text-align:center; vertical-align:middle'>16060</th>
      <th style='text-align:center; vertical-align:middle'>16061</th>
      <th style='text-align:center; vertical-align:middle'>16062</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>-73.0</td>
      <td style='text-align:center; vertical-align:middle'>-69.0</td>
      <td style='text-align:center; vertical-align:middle'>-48.0</td>
      <td style='text-align:center; vertical-align:middle'>13.0</td>
      <td style='text-align:center; vertical-align:middle'>-86.0</td>
      <td style='text-align:center; vertical-align:middle'>-147.0</td>
      <td style='text-align:center; vertical-align:middle'>-65.0</td>
      <td style='text-align:center; vertical-align:middle'>-71.0</td>
      <td style='text-align:center; vertical-align:middle'>-32.0</td>
      <td style='text-align:center; vertical-align:middle'>100.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>-134.0</td>
      <td style='text-align:center; vertical-align:middle'>352.0</td>
      <td style='text-align:center; vertical-align:middle'>-67.0</td>
      <td style='text-align:center; vertical-align:middle'>121.0</td>
      <td style='text-align:center; vertical-align:middle'>-5.0</td>
      <td style='text-align:center; vertical-align:middle'>-11.0</td>
      <td style='text-align:center; vertical-align:middle'>-21.0</td>
      <td style='text-align:center; vertical-align:middle'>-41.0</td>
      <td style='text-align:center; vertical-align:middle'>-967.0</td>
      <td style='text-align:center; vertical-align:middle'>-120.0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>-16.0</td>
      <td style='text-align:center; vertical-align:middle'>-63.0</td>
      <td style='text-align:center; vertical-align:middle'>-97.0</td>
      <td style='text-align:center; vertical-align:middle'>-42.0</td>
      <td style='text-align:center; vertical-align:middle'>-91.0</td>
      <td style='text-align:center; vertical-align:middle'>-164.0</td>
      <td style='text-align:center; vertical-align:middle'>-53.0</td>
      <td style='text-align:center; vertical-align:middle'>-77.0</td>
      <td style='text-align:center; vertical-align:middle'>-17.0</td>
      <td style='text-align:center; vertical-align:middle'>122.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>-51.0</td>
      <td style='text-align:center; vertical-align:middle'>244.0</td>
      <td style='text-align:center; vertical-align:middle'>-15.0</td>
      <td style='text-align:center; vertical-align:middle'>119.0</td>
      <td style='text-align:center; vertical-align:middle'>-32.0</td>
      <td style='text-align:center; vertical-align:middle'>4.0</td>
      <td style='text-align:center; vertical-align:middle'>-14.0</td>
      <td style='text-align:center; vertical-align:middle'>-28.0</td>
      <td style='text-align:center; vertical-align:middle'>-205.0</td>
      <td style='text-align:center; vertical-align:middle'>-31.0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>4.0</td>
      <td style='text-align:center; vertical-align:middle'>-45.0</td>
      <td style='text-align:center; vertical-align:middle'>-112.0</td>
      <td style='text-align:center; vertical-align:middle'>-25.0</td>
      <td style='text-align:center; vertical-align:middle'>-85.0</td>
      <td style='text-align:center; vertical-align:middle'>-127.0</td>
      <td style='text-align:center; vertical-align:middle'>56.0</td>
      <td style='text-align:center; vertical-align:middle'>-110.0</td>
      <td style='text-align:center; vertical-align:middle'>81.0</td>
      <td style='text-align:center; vertical-align:middle'>41.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>14.0</td>
      <td style='text-align:center; vertical-align:middle'>163.0</td>
      <td style='text-align:center; vertical-align:middle'>-14.0</td>
      <td style='text-align:center; vertical-align:middle'>7.0</td>
      <td style='text-align:center; vertical-align:middle'>15.0</td>
      <td style='text-align:center; vertical-align:middle'>-8.0</td>
      <td style='text-align:center; vertical-align:middle'>-104.0</td>
      <td style='text-align:center; vertical-align:middle'>-36.0</td>
      <td style='text-align:center; vertical-align:middle'>-245.0</td>
      <td style='text-align:center; vertical-align:middle'>34.0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>3</th>
      <td style='text-align:center; vertical-align:middle'>-31.0</td>
      <td style='text-align:center; vertical-align:middle'>-110.0</td>
      <td style='text-align:center; vertical-align:middle'>-20.0</td>
      <td style='text-align:center; vertical-align:middle'>-50.0</td>
      <td style='text-align:center; vertical-align:middle'>-115.0</td>
      <td style='text-align:center; vertical-align:middle'>-113.0</td>
      <td style='text-align:center; vertical-align:middle'>-17.0</td>
      <td style='text-align:center; vertical-align:middle'>-40.0</td>
      <td style='text-align:center; vertical-align:middle'>-17.0</td>
      <td style='text-align:center; vertical-align:middle'>80.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>26.0</td>
      <td style='text-align:center; vertical-align:middle'>625.0</td>
      <td style='text-align:center; vertical-align:middle'>18.0</td>
      <td style='text-align:center; vertical-align:middle'>59.0</td>
      <td style='text-align:center; vertical-align:middle'>-10.0</td>
      <td style='text-align:center; vertical-align:middle'>32.0</td>
      <td style='text-align:center; vertical-align:middle'>-2.0</td>
      <td style='text-align:center; vertical-align:middle'>10.0</td>
      <td style='text-align:center; vertical-align:middle'>-495.0</td>
      <td style='text-align:center; vertical-align:middle'>-37.0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>4</th>
      <td style='text-align:center; vertical-align:middle'>-33.0</td>
      <td style='text-align:center; vertical-align:middle'>-39.0</td>
      <td style='text-align:center; vertical-align:middle'>-45.0</td>
      <td style='text-align:center; vertical-align:middle'>14.0</td>
      <td style='text-align:center; vertical-align:middle'>-56.0</td>
      <td style='text-align:center; vertical-align:middle'>-106.0</td>
      <td style='text-align:center; vertical-align:middle'>73.0</td>
      <td style='text-align:center; vertical-align:middle'>-34.0</td>
      <td style='text-align:center; vertical-align:middle'>18.0</td>
      <td style='text-align:center; vertical-align:middle'>64.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>-69.0</td>
      <td style='text-align:center; vertical-align:middle'>398.0</td>
      <td style='text-align:center; vertical-align:middle'>38.0</td>
      <td style='text-align:center; vertical-align:middle'>215.0</td>
      <td style='text-align:center; vertical-align:middle'>-2.0</td>
      <td style='text-align:center; vertical-align:middle'>44.0</td>
      <td style='text-align:center; vertical-align:middle'>3.0</td>
      <td style='text-align:center; vertical-align:middle'>68.0</td>
      <td style='text-align:center; vertical-align:middle'>-293.0</td>
      <td style='text-align:center; vertical-align:middle'>-34.0</td>
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
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
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
      <th style='text-align:center; vertical-align:middle'>139</th>
      <td style='text-align:center; vertical-align:middle'>-196.0</td>
      <td style='text-align:center; vertical-align:middle'>-369.0</td>
      <td style='text-align:center; vertical-align:middle'>-263.0</td>
      <td style='text-align:center; vertical-align:middle'>162.0</td>
      <td style='text-align:center; vertical-align:middle'>-277.0</td>
      <td style='text-align:center; vertical-align:middle'>-615.0</td>
      <td style='text-align:center; vertical-align:middle'>-397.0</td>
      <td style='text-align:center; vertical-align:middle'>-243.0</td>
      <td style='text-align:center; vertical-align:middle'>70.0</td>
      <td style='text-align:center; vertical-align:middle'>-167.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>-25.0</td>
      <td style='text-align:center; vertical-align:middle'>2674.0</td>
      <td style='text-align:center; vertical-align:middle'>171.0</td>
      <td style='text-align:center; vertical-align:middle'>1499.0</td>
      <td style='text-align:center; vertical-align:middle'>95.0</td>
      <td style='text-align:center; vertical-align:middle'>735.0</td>
      <td style='text-align:center; vertical-align:middle'>-12.0</td>
      <td style='text-align:center; vertical-align:middle'>647.0</td>
      <td style='text-align:center; vertical-align:middle'>-2414.0</td>
      <td style='text-align:center; vertical-align:middle'>-33.0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>140</th>
      <td style='text-align:center; vertical-align:middle'>34.0</td>
      <td style='text-align:center; vertical-align:middle'>-81.0</td>
      <td style='text-align:center; vertical-align:middle'>-146.0</td>
      <td style='text-align:center; vertical-align:middle'>-151.0</td>
      <td style='text-align:center; vertical-align:middle'>-174.0</td>
      <td style='text-align:center; vertical-align:middle'>-121.0</td>
      <td style='text-align:center; vertical-align:middle'>-290.0</td>
      <td style='text-align:center; vertical-align:middle'>-106.0</td>
      <td style='text-align:center; vertical-align:middle'>43.0</td>
      <td style='text-align:center; vertical-align:middle'>240.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>-32.0</td>
      <td style='text-align:center; vertical-align:middle'>226.0</td>
      <td style='text-align:center; vertical-align:middle'>189.0</td>
      <td style='text-align:center; vertical-align:middle'>310.0</td>
      <td style='text-align:center; vertical-align:middle'>-13.0</td>
      <td style='text-align:center; vertical-align:middle'>210.0</td>
      <td style='text-align:center; vertical-align:middle'>-22.0</td>
      <td style='text-align:center; vertical-align:middle'>622.0</td>
      <td style='text-align:center; vertical-align:middle'>-889.0</td>
      <td style='text-align:center; vertical-align:middle'>-104.0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>141</th>
      <td style='text-align:center; vertical-align:middle'>-56.0</td>
      <td style='text-align:center; vertical-align:middle'>-818.0</td>
      <td style='text-align:center; vertical-align:middle'>-1338.0</td>
      <td style='text-align:center; vertical-align:middle'>-57.0</td>
      <td style='text-align:center; vertical-align:middle'>-989.0</td>
      <td style='text-align:center; vertical-align:middle'>-796.0</td>
      <td style='text-align:center; vertical-align:middle'>-1466.0</td>
      <td style='text-align:center; vertical-align:middle'>-347.0</td>
      <td style='text-align:center; vertical-align:middle'>-413.0</td>
      <td style='text-align:center; vertical-align:middle'>103.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>-85.0</td>
      <td style='text-align:center; vertical-align:middle'>1827.0</td>
      <td style='text-align:center; vertical-align:middle'>581.0</td>
      <td style='text-align:center; vertical-align:middle'>1547.0</td>
      <td style='text-align:center; vertical-align:middle'>-72.0</td>
      <td style='text-align:center; vertical-align:middle'>999.0</td>
      <td style='text-align:center; vertical-align:middle'>-461.0</td>
      <td style='text-align:center; vertical-align:middle'>564.0</td>
      <td style='text-align:center; vertical-align:middle'>-3567.0</td>
      <td style='text-align:center; vertical-align:middle'>-192.0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>142</th>
      <td style='text-align:center; vertical-align:middle'>-245.0</td>
      <td style='text-align:center; vertical-align:middle'>-235.0</td>
      <td style='text-align:center; vertical-align:middle'>-127.0</td>
      <td style='text-align:center; vertical-align:middle'>197.0</td>
      <td style='text-align:center; vertical-align:middle'>-562.0</td>
      <td style='text-align:center; vertical-align:middle'>-714.0</td>
      <td style='text-align:center; vertical-align:middle'>-1621.0</td>
      <td style='text-align:center; vertical-align:middle'>-226.0</td>
      <td style='text-align:center; vertical-align:middle'>-35.0</td>
      <td style='text-align:center; vertical-align:middle'>-243.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>-419.0</td>
      <td style='text-align:center; vertical-align:middle'>580.0</td>
      <td style='text-align:center; vertical-align:middle'>233.0</td>
      <td style='text-align:center; vertical-align:middle'>1065.0</td>
      <td style='text-align:center; vertical-align:middle'>-71.0</td>
      <td style='text-align:center; vertical-align:middle'>397.0</td>
      <td style='text-align:center; vertical-align:middle'>-28.0</td>
      <td style='text-align:center; vertical-align:middle'>114.0</td>
      <td style='text-align:center; vertical-align:middle'>-3086.0</td>
      <td style='text-align:center; vertical-align:middle'>-16.0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>143</th>
      <td style='text-align:center; vertical-align:middle'>-26.0</td>
      <td style='text-align:center; vertical-align:middle'>-1595.0</td>
      <td style='text-align:center; vertical-align:middle'>-2085.0</td>
      <td style='text-align:center; vertical-align:middle'>-334.0</td>
      <td style='text-align:center; vertical-align:middle'>-455.0</td>
      <td style='text-align:center; vertical-align:middle'>-354.0</td>
      <td style='text-align:center; vertical-align:middle'>-482.0</td>
      <td style='text-align:center; vertical-align:middle'>196.0</td>
      <td style='text-align:center; vertical-align:middle'>114.0</td>
      <td style='text-align:center; vertical-align:middle'>45.0</td>
      <td style='text-align:center; vertical-align:middle'>...</td>
      <td style='text-align:center; vertical-align:middle'>-243.0</td>
      <td style='text-align:center; vertical-align:middle'>526.0</td>
      <td style='text-align:center; vertical-align:middle'>126.0</td>
      <td style='text-align:center; vertical-align:middle'>320.0</td>
      <td style='text-align:center; vertical-align:middle'>-30.0</td>
      <td style='text-align:center; vertical-align:middle'>308.0</td>
      <td style='text-align:center; vertical-align:middle'>-179.0</td>
      <td style='text-align:center; vertical-align:middle'>121.0</td>
      <td style='text-align:center; vertical-align:middle'>-1878.0</td>
      <td style='text-align:center; vertical-align:middle'>-357.0</td>
    </tr>
  </tbody>
</table>
<p>144 rows × 16063 columns</p>
</div>

## **Clustering Analysis**

<h4>4. Use the<kbd>sklearn.cluster</kbd> module to perform clustering analysis on the dataset. In particular, repeat the analysis proposed in section 14.3.8 of the book (Hastie et al., 2009)</h4>

* Start using **`K-means`** and then test some **other clustering algorithms** at your choice
* Cluster the samples (i.e., columns). Each sample has a label (tumor type)
* **Do not use the labels in the clustering phase** but examine them posthoc to interpret the clusters
* Run k-means with K from **2 to 10** and compare the clusterings in terms of within-sum of squares
* Show the chart of the performance depending on K
* Select some K and analyze the clusters as done in the book

### **K-Means**
The **KMeans algorithm** clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the **inertia** or **within-cluster sum-of-squares**.<cite> [^1]</cite>
[^1]: More info about K-Means method and other clustering methods can be found [here](https://scikit-learn.org/stable/modules/clustering.html)

The k-means algorithm divides a set of samples into disjoint clusters, each described by the mean of the samples in the cluster. The means are commonly called the cluster “*centroids*”; note that they are not, in general, points from, although they live in the same space.

The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:
\\[\\sum\_{i=0}^{n}\\min\_{\\mu\_j \\in C}(||x\_i - \\mu\_j||^2)\\]


```python
# K-means with k from 2 to 10

n_clusters = range(2,11)
alg = 'k-means++' # Method for initialization
niter = 10 # Number of time the k-means algorithm will be run with different centroid seeds.
wc_km_sos=[] # within_cluster_km_sos

print('k\tInertia\t\t\tdecrease %')
print(50 * '-')
formatter_result = ("{:d}\t{:f}\t{:f}")

for k in n_clusters:
    
    results = []
    results.append(k)
    
    km = KMeans(init=alg, n_clusters=k, n_init=niter).fit(xtrain)
    
    # inertia = Sum of squared distances of samples to their closest cluster center  
    wcv = km.inertia_
    wc_km_sos.append(wcv)
    results.append(wcv)
    
    # variations in %
    if len(wc_km_sos)>1:
        results.append(
            (wc_km_sos[k-2] - wc_km_sos[k-3])*100/wc_km_sos[k-2]
        )
    else:
        results.append(0)
        
    print(formatter_result.format(*results))    
```

    k	Inertia			        decrease %
    --------------------------------------------------
    2	865755593329.079102 	0.000000
    3	728028390816.590332	    -18.917834
    4	638452947540.124023	    -14.030077
    5	586449466492.984497	    -8.867513
    6	538128754493.843750 	-8.979396
    7	516487727616.067871 	-4.190037
    8	488855618252.548340 	-5.652407
    9	454592949466.325562	    -7.537000
    10	440415484775.366333	    -3.219111



```python
# fig 
width, height = 8, 4
fig, ax = plt.subplots(figsize=(width,height))

ax.plot(n_clusters, wc_km_sos,  marker='o', color="darkblue")
ax.grid(color='grey', linestyle='-', linewidth=0.5);
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))

ax.set_xlabel("Number of Clusters K", fontsize=12)
ax.set_ylabel("within-cluster sum of squares", fontsize=12)

plt.suptitle("Total within-cluster sum of squares\n for K-means clustering",fontsize=20)
plt.subplots_adjust(top=0.825) # change title position

plt.show()
```


![png](/posts/sl-ex5-humantumormicro-clustering/output_15_0.png)



```python
# We can compare the above chart with the one in the book:
scale = 80
Image("../input/14cancer/chart.png", width = width*scale, height = height*scale)
```




![png](/posts/sl-ex5-humantumormicro-clustering/output_16_0.png)
This plot is taken from "The Elements of Statistical Learning" book.<cite> [^2]</cite>
[^2]: The above chart can be found in the section 14.3.8 of the book [The Elements of Statistical Learning: Data Mining, Inference, and Prediction.](https://hastie.su.domains/ElemStatLearn/)


### **Comparison between different methods of initialization: <kbd>k-means++</kbd> vs <kbd>random</kbd>**


```python
n_clusters = range(2,11)
niter = 10 # Number of time the k-means algorithm will be run with different centroid seeds.
wc_kmpp, wc_rnd = [], []

print('k\tK-means\t\t\trandom')
print(60 * '-')
formatter_result = ("{:d}\t{:f}\t{:f}")

for k in n_clusters:
    
    results = []
    results.append(k)
    
    kmpp = KMeans(init="k-means++", n_clusters=k, n_init=niter).fit(xtrain)
    rnd = KMeans(init="random", n_clusters=k, n_init=niter).fit(xtrain)

    results.append(kmpp.inertia_)
    results.append(rnd.inertia_)
   
    wc_kmpp.append(kmpp.inertia_)
    wc_rnd.append(rnd.inertia_)

    print(formatter_result.format(*results))    
```

    k	K-means			        random
    ------------------------------------------------------------
    2	865755593329.079102	    865755593329.078979
    3	728215342983.054443	    728215342983.054443
    4	638286863470.537109	    638452947540.124146
    5	586098738159.229004	    580943572411.067993
    6	541362331453.668213	    539591832514.661987
    7	501565429046.019531	    500472648214.279541
    8	481877683922.631714	    484882990782.917847
    9	461806611237.345337	    464195618439.327515
    10	448128965453.922974	    454970652718.346436



```python
# fig 
width, height = 8, 4
fig, ax = plt.subplots(figsize=(width,height))

ax.plot(n_clusters, wc_kmpp,  marker='*', color="darkblue", label = "k-means++")
ax.plot(n_clusters, wc_rnd,  marker='o', color="orange", label = "random")

ax.grid(color='grey', linestyle='-', linewidth=0.5);
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.legend()
ax.set_xlabel("Number of Clusters K", fontsize=12)
ax.set_ylabel("within-cluster sum of squares", fontsize=12)

plt.suptitle("Comparison with different method of initialization",fontsize=20)
plt.subplots_adjust(top=0.825) # change title position

plt.show()
```


![png](/posts/sl-ex5-humantumormicro-clustering/output_19_0.png)


### **Comparison between different <kbd>n_iter</kbd>**:
Number of time the k-means algorithm will be run with different centroid seeds.


```python
n_clusters = range(2,11)
wc_ten_seeds, wc_twenty_seeds = [], []

print('k\tn_iter=10\t\tn_iter=20')
print(70 * '-')
formatter_result = ("{:d}\t{:f}\t{:f}")

for k in n_clusters:
    
    results = []
    results.append(k)
    
    ten_seeds = KMeans(init="k-means++", n_clusters=k, n_init=10).fit(xtrain)
    twenty_seeds = KMeans(init="k-means++", n_clusters=k, n_init=20).fit(xtrain)

    results.append(ten_seeds.inertia_)
    results.append(twenty_seeds.inertia_)
   
    wc_ten_seeds.append(ten_seeds.inertia_)
    wc_twenty_seeds.append(twenty_seeds.inertia_)

    print(formatter_result.format(*results))    
```

    k	n_iter=10		        n_iter=20
    ----------------------------------------------------------------------
    2	866070488704.476074	    865755593329.079102
    3	728028390816.590210	    727972625271.491211
    4	639723868185.660278 	638286863470.537109
    5	579977474766.300903 	580224574540.047607
    6	543140308602.200195 	537625894944.809998
    7	499824352123.143555 	499900728332.191284
    8	481177796841.305420 	478729684111.517700
    9	463786737203.969238 	455823165084.713989
    10	447920765947.759399 	440614709199.603394



```python
# fig 
width, height = 8, 4
fig, ax = plt.subplots(figsize=(width,height))

ax.plot(n_clusters, wc_ten_seeds,  marker='*', color="darkblue", label = "n_iter=10")
ax.plot(n_clusters, wc_twenty_seeds,  marker='o', color="orange", label = "n_iter=20")

ax.grid(color='grey', linestyle='-', linewidth=0.5);
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.legend()
ax.set_xlabel("Number of Clusters K", fontsize=12)
ax.set_ylabel("within-cluster sum of squares", fontsize=12)

plt.suptitle("Comparison between different n_iter",fontsize=20)
plt.subplots_adjust(top=0.825) # change title position

plt.show()
```


![png](/posts/sl-ex5-humantumormicro-clustering/output_22_0.png)


### **Mini-batch K-means**
The **MiniBatchKMeans** is a variant of the KMeans algorithm which uses mini-batches to reduce the computation time, while still attempting to optimise the same objective function.

Mini-batches are subsets of the input data, randomly sampled in each training iteration. These mini-batches drastically reduce the amount of computation required to converge to a local solution.

In contrast to other algorithms that reduce the convergence time of k-means, mini-batch k-means produces results that are generally only slightly worse than the standard algorithm.<cite> [^1]</cite>


```python
# K-means with k from 2 to 10

n_clusters = range(2,11)
alg = 'k-means++' # Method for initialization
niter = 10 # Number of time the k-means algorithm will be run with different centroid seeds.
wc_mbkm_sos=[]

print('k\tInertia\t\t\tdecrease %')
print(50 * '-')
formatter_result = ("{:d}\t{:f}\t{:f}")

for k in n_clusters:
    
    results = []
    results.append(k)
    
    mbkm = MiniBatchKMeans(init=alg, n_clusters=k, n_init=niter).fit(xtrain)
    
    # inertia = Sum of squared distances of samples to their closest cluster center  
    wcv = mbkm.inertia_
    wc_mbkm_sos.append(wcv)
    results.append(wcv)
    
    # variations in %
    if len(wc_mbkm_sos)>1:
        results.append(
            (wc_mbkm_sos[k-2] - wc_mbkm_sos[k-3])*100/wc_mbkm_sos[k-2]
        )
    else:
        results.append(0)
        
    print(formatter_result.format(*results))    

```
    k	Inertia			        decrease %
    --------------------------------------------------
    2	870435631688.268311	    0.000000
    3	728979505913.590698	    -19.404678
    4	644499761950.796875	    -13.107801
    5	651489332987.863525	    1.072860
    6	554034243741.206177	    -17.590084
    7	547550479471.734985	    -1.184140
    8	526030833538.899902	    -4.090948
    9	489328464969.691284	    -7.500559
    10	452845818672.533325	    -8.056306



```python
# fig 
width, height = 8, 4
fig, ax = plt.subplots(figsize=(width,height))

ax.plot(n_clusters, wc_mbkm_sos,  marker='o', color="darkblue", label = "k-means")
ax.plot(n_clusters, wc_km_sos,  marker='o', color="orange",  label = "Mini batch k-means")
ax.grid(color='grey', linestyle='-', linewidth=0.5);
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))

ax.legend()
ax.set_xlabel("Number of Clusters K", fontsize=12)
ax.set_ylabel("within-cluster sum of squares", fontsize=12)

plt.suptitle("Total within-cluster sum of squares\ncomparison",fontsize=20)
plt.subplots_adjust(top=0.825) # change title position

plt.show()
```


![png](/posts/sl-ex5-humantumormicro-clustering/output_29_0.png)



## **Analysis for K=3**

### **Number of cancer cases of each type in each of the 3 clusters**


```python
rows = KMeans(init="k-means++", n_clusters=3).fit(xtrain).labels_ # labels of each sample after clustering
columns = ytrain.to_numpy().flatten() # make the df into an iterable list

# Collect info in a table
tab = np.zeros(3*n_labels).reshape(3,n_labels) # rows: clusters, columns: cancer labels

# Update table
for i in range(n_samples):
    tab[rows[i],columns[i]-1]+=1 # column-1 because we range over 14 clusters (0,13)
    
# Better formatting of the table into a DataFrame
table = pd.DataFrame(tab.astype(int))
table.columns = ["breast", "prostate", "lung", "collerectal", "lymphoma", "bladder",
                 "melanoma", "uterus", "leukemia", "renal", "pancreas", "ovary", "meso", "cns"]
```


```python
table
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
      <th style='text-align:center; vertical-align:middle'>breast</th>
      <th style='text-align:center; vertical-align:middle'>prostate</th>
      <th style='text-align:center; vertical-align:middle'>lung</th>
      <th style='text-align:center; vertical-align:middle'>collerectal</th>
      <th style='text-align:center; vertical-align:middle'>lymphoma</th>
      <th style='text-align:center; vertical-align:middle'>bladder</th>
      <th style='text-align:center; vertical-align:middle'>melanoma</th>
      <th style='text-align:center; vertical-align:middle'>uterus</th>
      <th style='text-align:center; vertical-align:middle'>leukemia</th>
      <th style='text-align:center; vertical-align:middle'>renal</th>
      <th style='text-align:center; vertical-align:middle'>pancreas</th>
      <th style='text-align:center; vertical-align:middle'>ovary</th>
      <th style='text-align:center; vertical-align:middle'>meso</th>
      <th style='text-align:center; vertical-align:middle'>cns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style='text-align:center; vertical-align:middle'>0</th>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>21</td>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>1</th>
      <td style='text-align:center; vertical-align:middle'>8</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>7</td>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>5</td>
      <td style='text-align:center; vertical-align:middle'>6</td>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
    </tr>
    <tr>
      <th style='text-align:center; vertical-align:middle'>2</th>
      <td style='text-align:center; vertical-align:middle'>0</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>13</td>
      <td style='text-align:center; vertical-align:middle'>1</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>3</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>2</td>
      <td style='text-align:center; vertical-align:middle'>4</td>
      <td style='text-align:center; vertical-align:middle'>13</td>
    </tr>
  </tbody>
</table>
</div>


