---
title: "Sampling distribution of Hosmer-Lemeshow statistic"
author: "Hamed Helali"
date: 2020-02-01
tags: [GLM, Goodness of fit statistic, Hosmer-Lemeshow, R]
categories: Blog-post
header:
  image: "/images/asian.jpg"
excerpt: "GLM, Goodness of fit statistic, Hosmer-Lemeshow, R"
mathjax: "true"
---

Dobson and Barnet(2008) in their generalized linear models (GLM) textbook, say:
> The sampling distribution of $$X_{HL}$$ has been found by simulation to be approximately χ2(g − 2).  

In this post I check out this statement:


```r
library(generalhoslem)
```



{% highlight text %}
## Loading required package: reshape
{% endhighlight %}



{% highlight text %}
## Loading required package: MASS
{% endhighlight %}



```r
set.seed(50)
NREP = 500
n_obs = 100

statistic_values <- array(0,NREP)
for (i in 1:NREP){
  x <- rnorm(n_obs, 1, 1)
  beta <- c(0.1,0.2)
  y <- rbinom(n_obs, size = 1, prob = (1/(1+exp(-(beta[1]+beta[2]*x)))))
  fit4 <- glm(y ~ x, family = binomial())
  hoslem <- logitgof(y,fit4$fitted.values)
  statistic_values[i] = hoslem$statistic
}
mean(statistic_values)
```



{% highlight text %}
## [1] 8.168053
{% endhighlight %}



```r
var(statistic_values)
```



{% highlight text %}
## [1] 13.93862
{% endhighlight %}



```r
qqplot(statistic_values, rchisq(500,8))
abline(0,1,lty=2)
```

![center](/images/2020-02-01-hoslem/unnamed-chunk-1-1.png)

This qqpolt compares values of the statistic with chi-square(df=8) and we can confirm that this statistic has distribution of chi-square with (g-2) degrees of freedom.
