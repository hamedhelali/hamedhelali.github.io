---
title: "Variance reduction methods for option pricing with Monte Carlo simulation"
author: "Hamed Helali"
date: 2020-11-02
tags: [Variance reduction, Monte Carlo simulation, Asian option pricing, R]
categories: Blog-post
header:
  image: "/images/asian.jpg"
excerpt: "Variance reduction, Monte Carlo simulation, Asian option pricing, R"
mathjax: "true"
---

In this post, we assume a hypothetical (asian) option with the following
specifications and try different varience reduction methods when
employing Monte Carlo simulation for pricing this option.

``` r
S_0 = 50
K = 52
sigma = 0.5
r = 0.05
N = 30
```

Before starting, I define a function that returns estimation, 95%
confidence interval and its length:

``` r
estimate = function(vector){
  NREP = length(vector)
  est = mean(vector)
  lower_b = est - sd(vector) * qnorm(0.975) / sqrt(NREP)
  upper_b = est + sd(vector) * qnorm(0.975) / sqrt(NREP)
  len = upper_b - lower_b
  return(c(est, lower_b, upper_b, len))
}
```

a) European option pricing
==========================

First I define a price calculator function which calculates payoff at
time T and then discounts it back to the present time.

``` r
price_calc = function(S_T, t, K, r){
  library(Rfast)
  payoff = rowMaxs(cbind(0, S_T - K), value = TRUE)
  return (exp(-r * t / 365) * payoff)
}
```

``` r
set.seed(16)
NREP = 1000
Z = rnorm(NREP)
S_T = S_0 * exp((r - 0.5 * sigma^2) * N / 365 + sigma * Z * sqrt(N / 365))

current_price = price_calc(S_T, N, K, r)
```


``` r
est = estimate(current_price)
paste("The fair price for this option is:", est[1])
```

    ## [1] "The fair price for this option is: 2.14767605028461"

``` r
paste("Confidence interval for the estimation is: (",est[2], ",", est[3], ")")
```

    ## [1] "Confidence interval for the estimation is: ( 1.90133340831646 , 2.39401869225276 )"

b) Asian Option Pricing (using naive MC)
========================================

``` r
asian_price_calc = function(price_path, t, K, r){
  library(Rfast)
  s_bar = mean(price_path)
  payoff = rowMaxs(cbind(0, s_bar - K), value = TRUE)
  return (exp(-r * t / 365) * payoff)
}
```

``` r
set.seed(5)
NREP = 1000
steps = 30
price_nmc = c()
Z = matrix(rnorm(NREP*steps), NREP, steps)

for (i in 1:NREP){
  price_path = c(S_0)
  for (j in 1:steps){
    new_price = price_path[j] * exp((r - 0.5 * sigma^2) / 365 + sigma * Z[i,j] * sqrt(1 / 365))
    price_path = c(price_path, new_price)
  }
  p = asian_price_calc(price_path, 30, K, r)
  price_nmc = c(price_nmc, p)
}
```

``` r
est = estimate(price_nmc)
paste("The fair price for this option is:", est[1])
```

    ## [1] "The fair price for this option is: 0.90531167321582"

``` r
paste("Confidence interval for the estimation is: (",est[2], ",", est[3], ")")
```

    ## [1] "Confidence interval for the estimation is: ( 0.791689961310881 , 1.01893338512076 )"

``` r
results = est
```

c) Control Variate for Asian Option Pricing
===========================================

We choose geometric-average asian call option as the control variate for
the arithmatic-average asian call option. We call the geometric asian
option price *θ* and we can analytically obtain it by (6.79) in the
textbook and I will implement the computing function here:

``` r
exact_geo_asian = function(S_0, K, t, r, sigma, N){
  c3 = 1 + 1/N
  c2 = sigma * ((c3 * t / 1095) * (1 + 1 / (2 * N))) ^ 0.5
  c1 = (1 / c2) * (log(S_0 / K) + (c3 * t / 730) * (r - 0.5 * sigma^2) + (c3 * sigma^2 * t / 1095) * (1 + 1 / (2 * N)))
  
  theta = S_0 * pnorm(c1) * exp(-t * (r + c3 * sigma^2 / 6) * (1 - 1/N) / 730) - K * pnorm(c1 - c2) * exp(-r * t / 365)
  return(theta)
}
```

So, the exact price of geometric asian option is:

``` r
theta = exact_geo_asian(S_0, K, t=30, r, sigma, N)
theta
```

    ## [1] 0.9096794

So, $$ \theta = 0.9097 $$. Now, control variate estimation of arithmatic asian
option can be obtained by:

$$
\hat \mu_{CV} = \hat \mu_{MC} + \lambda (\hat \theta_{MC} - \theta)
$$

Where the best choice for *λ* (which decreases the variance most) is:

$$
\lambda = \frac{-cov(\hat{\mu}_{MC}, \hat\theta_{MC})}{var(\hat\theta\{MC})}
$$

 And the covarience and varience need to be estimated using
simulation.  
We also need to simulate geometric asian option:

``` r
geo_asian_price_calc = function(price_path, t, K, r){
  library(Rfast)
  s_tilda = exp(mean(log(price_path)))
  payoff = rowMaxs(cbind(0, s_tilda - K), value = TRUE)
  return (exp(-r * t / 365) * payoff)
}
```

``` r
set.seed(5)
NREP = 1000
steps = 30
g_price_nmc = c()
Z = matrix(rnorm(NREP*steps), NREP, steps)

for (i in 1:NREP){
  price_path = c(S_0)
  for (j in 1:steps){
    new_price = price_path[j] * exp((r - 0.5 * sigma^2) / 365 + sigma * Z[i,j] * sqrt(1 / 365))
    price_path = c(price_path, new_price)
  }
  p = geo_asian_price_calc(price_path, 30, K, r)
  g_price_nmc = c(g_price_nmc, p)
}
mean(g_price_nmc)
```

    ## [1] 0.8691086

Therefore, $$ \hat \theta_{MC} = 0.8691 $$. Now, we estimate $$ \lambda $$ as
follows:

``` r
est_cov = sum((price_nmc - mean(price_nmc)) * (g_price_nmc - mean(g_price_nmc))) / (NREP * (NREP - 1))
est_var = sum((g_price_nmc - mean(g_price_nmc))^2) / (NREP * (NREP - 1))
lambda = - est_cov / est_var
lambda
```

    ## [1] -1.035134

``` r
CV_prices = price_nmc + lambda * (g_price_nmc - theta)

est = estimate(CV_prices)
paste("The fair price for this option is:", est[1])
```

    ## [1] "The fair price for this option is: 0.94730790041362"

``` r
paste("Confidence interval for the estimation is: (",est[2], ",", est[3], ")")
```

    ## [1] "Confidence interval for the estimation is: ( 0.944336365429991 , 0.95027943539725 )"

``` r
results = rbind(results, est)
```

d) Antithetic method for Asian Option Pricing
=============================================

``` r
set.seed(345)
NREP = 1000
steps = 30
anti_price1 = c()
anti_price2 = c()
Z = matrix(rnorm(NREP*steps), NREP, steps)

for (i in 1:NREP){
  price_path1 = c(S_0)
  price_path2 = c(S_0)
  for (j in 1:steps){
    new_price1 = price_path1[j] * exp((r - 0.5 * sigma^2) / 365 + sigma * Z[i,j] * sqrt(1 / 365))
    price_path1 = c(price_path1, new_price1)
    
    new_price2 = price_path2[j] * exp((r - 0.5 * sigma^2) / 365 - sigma * Z[i,j] * sqrt(1 / 365))
    price_path2 = c(price_path2, new_price2)
  }
  p1 = asian_price_calc(price_path1, 30, K, r)
  anti_price1 = c(anti_price1, p1)
  
  p2 = asian_price_calc(price_path2, 30, K, r)
  anti_price2 = c(anti_price2, p2)
}
```

``` r
anti_price = 0.5 * (anti_price1 + anti_price2)
est = estimate(anti_price)
paste("The fair price for this option is:", est[1])
```

    ## [1] "The fair price for this option is: 0.895692687977585"

``` r
paste("Confidence interval for the estimation is: (",est[2], ",", est[3], ")")
```

    ## [1] "Confidence interval for the estimation is: ( 0.821176444513217 , 0.970208931441953 )"

``` r
results = rbind(results, est)
```

e) Comparing the methods
========================

Results for arithmatic-average asian call option using different methods
of MC are as follows:

``` r
rownames(results) = c("Naive MC", "Control variate MC", "Antithetic MC")
colnames(results) = c("Price estimation", "CI lower bound", "CI upper bound", "CI length")
results
```

    ##                    Price estimation CI lower bound CI upper bound  CI length
    ## Naive MC                  0.9053117      0.7916900      1.0189334 0.22724342
    ## Control variate MC        0.9473079      0.9443364      0.9502794 0.00594307
    ## Antithetic MC             0.8956927      0.8211764      0.9702089 0.14903249

``` r
library(dotwhisker)
```

    ## Loading required package: ggplot2

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:Rfast':
    ## 
    ##     nth

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(broom)

NREP = 1000

term = c("Naive MC", "Control variate MC", "Antithetic MC")
estim = results[,1]
std = c(sd(price_nmc) / sqrt(NREP), sd(CV_prices) / sqrt(NREP), sd(anti_price) / sqrt(NREP))
std
```

    ## [1] 0.057971326 0.001516117 0.038019190

``` r
results_df = data.frame(term=term, estimate=estim, std.error=std)

dwplot(results_df)
```

![](/unnamed-chunk-18-1.png)

It is observable that control variate method provides much smaller
variance. However, it seems that the estimated price is a little bit
biased and I could not figure out why.
