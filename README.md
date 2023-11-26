# Bootstrap-CI-analysis
**Comparison of different methods for confidence interval calculation.**

# Table of contents

# Abstract
Quantifying uncertainty is a vital part of every statistical study. There are many different methods, but in the hands 
of an inexperienced user, most of them can lead to big mistakes in the interpretation. 
Bootstrap is a favorable method for this task because of its robustness, versatility, ease of understanding and lack of 
stringent distributional assumptions. 
But even after 40 years of existence, it is not clear if this general method is accurate enough to substitute the 
traditional methods specialized for specific parameters of interest.
To answer this, we designed an extensive simulation study that assesses the methods' confidence interval prediction for 
six different parameters on samples of multiple sizes, generated from seven diverse distributions. 
We chose the double bootstrap as the best general bootstrap method, additionally recommending the standard bootstrap 
for confidence intervals for extreme percentiles.
We compared the best bootstrap methods to the traditional methods and found out that for almost all of the parameters 
no traditional method is practically better. Moreover, bootstrap gives good predictions even on the distributions where 
traditional methods fail because of broken assumptions.
Our work thus suggests that estimates generated by the proposed bootstrap methods are comparable to or even better than 
the ones made by the traditional methods.

# Methods
We compared the most commonly used bootstrap methods to traditional methods for confidence interval
calculation. 
We then compared their accuracy and correctness, to see where choosing bootstrap over the traditional ones 
would be a mistake. 
For a more detailed description of methods for confidence interval estimation and experiment setup we refer the reader 
to the thesis available in folder `thesis`.

## Methods for calculation of confidence intervals
We used the bootstrap methods implemented in the `bootstrap-ci` library: *percentile*, *basic*, *standard*, *BC*, 
*BC_a*, *smoothed*, *studentized* and *double*. 
Their description can be seen in the library's
[repository](https://github.com/zrimseku/bootstrap-ci). 

## Experiment dimensions
To get the most general results possible, we compared all methods over many combinations of different DGP's, statistics, 
dataset sizes and coverage levels. 
Data generating processes used are implemented in the file `generators.py`, where you can also add your custom DGP, by
extending the `DGP` class. 
We used the following distributions:
- standard normal,
- uniform from $0$ to $1$,
- Laplace with $\mu = 0, b = 1$,
- beta with $\alpha = 2, \beta = 10$,
- exponential with $\lambda = 1$,
- log-normal with $\mu = 0, \sigma = 1$
- bi-normal with $\mu = 
            \begin{bmatrix}
            1 \\
            1
            \end{bmatrix}$ 
            and $\Sigma = 
            \begin{bmatrix}
            2 & 0.5\\
            0.5 & 1
            \end{bmatrix}$.
  
We used samples of sizes $n \in \{4, 8, 16, 32, 64, 128, 256\}$ randomly generated from these distributions to estimate 
confidence intervals for the *mean*, *median*, *standard deviation*, *5^\{th\}* and *95^{th} percentile* and *correlation*.
We were interested in confidence levels $\alpha \in {0.025, 0.05, 0.25, 0.75, 0.95, 0.975}$.

## Framework
To compare the methods we used two criteria: *accuracy* and *correctness*. Accuracy is the more important one, telling 
us how close the method's achieved coverage is to the desired one. If two methods achieve the same accuracy, we compared
their correctness, which is calculated by the distance of each method's predicted confidence intervals to the *exact* 
intervals.

The study was done in three steps:
1. Choosing the best bootstrap method.
2. Comparing the best bootstrap method to all other methods (bootstrap and traditional ones), to see where another 
method gives better one-sided confidence interval estimations.
3. Repeating step 2. for two-sided confidence intervals.

## Hierarchical bootstrap
We compared different strategies of the *cases* bootstrap based on their accuracy and ability to mimic the DGP's 
variational properties.

# Results
More detailed results can again be found in the thesis, in chapter 4.
In short, we answered to the above steps:
1. The best general bootstrap method is the *double* bootstrap. Additionally we recommend to use the *standard* bootstrap 
   when estimating confidence intervals of extreme percentiles.
2. There is no method (bootstrap or traditional) that would have significantly better accuracy in most of the 
   repetitions for experiments on any DGP. Only for the correlation, Fisher's method is equally accurate but more
   correct.
3. Results are similar for two-sided intervals.

## Hierarchical bootstrap
We recommend using the strategy that samples with replacement on all levels, as it has the best accuracy and it best
mimics the DGP's variational properties.

# Reproducibility and custom experiments