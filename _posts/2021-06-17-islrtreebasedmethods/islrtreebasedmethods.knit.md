---
title: "ISLR Machine Learning "
description: |
  A short description of the post.
author:
  - name: Jixing Liu
    url: https://emitanaka.org
    affiliation: Monash University
    affiliation_url: https://numbat.space/
date: 2021-02-03
draft: false
categories: [machine learning, R]
output:
  distill::distill_article:
    self_contained: false
    highlight: default
    toc: true
    #highlight_downlit: false # downlit makes attr.source not work
    toc_float: true
preview: "islrtreebasedmethods_photo/image-20210617171443567.png"
---





 ![](islrtreebasedmethods_photo/image-20210617171443567.png)



## load pkg



## load data

<div class="layout-chunk" data-layout="l-body">
<div class='toggle-code'><div class="sourceCode"><pre class="sourceCode r"><code class="sourceCode r"><span class='kw'><a href='https://rdrr.io/r/base/library.html'>library</a></span><span class='op'>(</span><span class='va'><a href='http://www.StatLearning.com'>ISLR</a></span><span class='op'>)</span>
<span class='kw'><a href='https://rdrr.io/r/base/library.html'>library</a></span><span class='op'>(</span><span class='va'><a href='http://www.stats.ox.ac.uk/pub/MASS4/'>MASS</a></span><span class='op'>)</span>
<span class='kw'><a href='https://rdrr.io/r/base/library.html'>library</a></span><span class='op'>(</span><span class='va'><a href='https://tidymodels.tidymodels.org'>tidymodels</a></span><span class='op'>)</span>
<span class='co'># Boston &lt;- as_tibble(Boston)</span>
<span class='va'>Carseats</span> <span class='op'>&lt;-</span> <span class='fu'>as_tibble</span><span class='op'>(</span><span class='va'>Carseats</span><span class='op'>)</span> <span class='op'>%&gt;%</span>
  <span class='fu'>mutate</span><span class='op'>(</span>High <span class='op'>=</span> <span class='fu'><a href='https://rdrr.io/r/base/factor.html'>factor</a></span><span class='op'>(</span><span class='fu'>if_else</span><span class='op'>(</span><span class='va'>Sales</span> <span class='op'>&lt;=</span> <span class='fl'>8</span>, <span class='st'>"No"</span>, <span class='st'>"Yes"</span><span class='op'>)</span><span class='op'>)</span><span class='op'>)</span>

<span class='va'>data</span> <span class='op'>&lt;-</span> 
<span class='va'>Carseats</span> <span class='op'>%&gt;%</span> 
  <span class='fu'>dplyr</span><span class='fu'>::</span><span class='fu'><a href='https://dplyr.tidyverse.org/reference/select.html'>select</a></span><span class='op'>(</span><span class='op'>-</span><span class='va'>Sales</span><span class='op'>)</span>

<span class='va'>data</span> <span class='op'>&lt;-</span> <span class='va'>data</span> <span class='op'>%&gt;%</span> 
  <span class='fu'>dplyr</span><span class='fu'>::</span><span class='fu'><a href='https://dplyr.tidyverse.org/reference/mutate.html'>mutate</a></span><span class='op'>(</span>
    High <span class='op'>=</span> <span class='fu'><a href='https://rdrr.io/r/base/factor.html'>as.factor</a></span><span class='op'>(</span><span class='va'>High</span><span class='op'>)</span>,
    High <span class='op'>=</span> <span class='fu'>fct_collapse</span><span class='op'>(</span><span class='va'>High</span>,
                        `0` <span class='op'>=</span> <span class='fu'><a href='https://rdrr.io/r/base/c.html'>c</a></span><span class='op'>(</span><span class='st'>"No"</span><span class='op'>)</span>,
                        `1` <span class='op'>=</span> <span class='fu'><a href='https://rdrr.io/r/base/c.html'>c</a></span><span class='op'>(</span><span class='st'>"Yes"</span><span class='op'>)</span><span class='op'>)</span>,
    High <span class='op'>=</span> <span class='fu'>fct_relevel</span><span class='op'>(</span><span class='va'>High</span>,
                       <span class='st'>"0"</span>,
                       <span class='st'>"1"</span><span class='op'>)</span><span class='op'>)</span>
</code></pre></div>
</div>

</div>


## split data
<div class="layout-chunk" data-layout="l-body">
<div class='toggle-code'><div class="sourceCode"><pre class="sourceCode r"><code class="sourceCode r"><span class='fu'><a href='https://rdrr.io/r/base/Random.html'>set.seed</a></span><span class='op'>(</span><span class='fl'>123</span><span class='op'>)</span>
<span class='va'>data_split</span> <span class='op'>&lt;-</span> <span class='fu'>initial_split</span><span class='op'>(</span><span class='va'>data</span>, prop <span class='op'>=</span> <span class='fl'>0.7</span><span class='op'>)</span>

<span class='va'>data_train</span> <span class='op'>&lt;-</span> <span class='fu'>training</span><span class='op'>(</span><span class='va'>data_split</span><span class='op'>)</span>
<span class='va'>data_test</span> <span class='op'>&lt;-</span> <span class='fu'>testing</span><span class='op'>(</span><span class='va'>data_split</span><span class='op'>)</span>
</code></pre></div>
</div>

</div>


# 🐍 module
<div class="layout-chunk" data-layout="l-body">
<div class='toggle-code'>

```python
import IPython
from IPython.display import HTML, display, Markdown, IFrame

from pycaret.classification import *
import pandas as pd               #data loading and manipulation
import matplotlib.pyplot as plt   #plotting
import seaborn as sns             #statistical plotting
```

</div>

</div>



## r data to python

<div class="layout-chunk" data-layout="l-body">
<div class='toggle-code'>

```python
train_data = r.data_train
test_data = r.data_test
train_data.head()
```

```
   CompPrice  Income  Advertising  Population  Price ShelveLoc   Age  \
0      104.0    71.0         14.0        89.0   81.0    Medium  25.0   
1      115.0    28.0         11.0        29.0   86.0      Good  53.0   
2      112.0    98.0         18.0       481.0  128.0    Medium  45.0   
3      115.0    29.0         26.0       394.0  132.0    Medium  33.0   
4      145.0    53.0          0.0       507.0  119.0    Medium  41.0   

   Education Urban   US High  
0       14.0    No  Yes    1  
1       18.0   Yes  Yes    1  
2       11.0   Yes  Yes    0  
3       13.0   Yes  Yes    1  
4       12.0   Yes   No    1  
```

</div>

</div>



## pycaret setup 

[Functions - PyCaret](https://pycaret.org/functions/)

























