---
output: html_document
---

To implement EM algorithm for probit regression, at first I am going to prepare simulated data for modeling.

{% highlight r %}
x <- rnorm(2000,0,2)
betas <- c(0.1 , 0.2) #True values
z <- rnorm(2000, mean = cbind(1,x) %*% betas, sd=1)
y <- z
y[y>=0] <- 1
y[y<0] <- 0
t <- glm(y~x, family = binomial(link = "probit"))
summary(t)
{% endhighlight %}



{% highlight text %}
## 
## Call:
## glm(formula = y ~ x, family = binomial(link = "probit"))
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.0901  -1.1201   0.6738   1.0640   1.8969  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  0.06236    0.02894   2.155   0.0312 *  
## x            0.21372    0.01547  13.820   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 2766.1  on 1999  degrees of freedom
## Residual deviance: 2562.3  on 1998  degrees of freedom
## AIC: 2566.3
## 
## Number of Fisher Scoring iterations: 4
{% endhighlight %}

As the next step, we should calculate $\gamma_i$:
$$
\gamma_i=E(z_i|y=y_i)
$$
$z_i$s conditioned on $y_i$ have truncated normal distribution. So the expected value can be calculated by the equations below. If $y_i=1$,
$$
E(z_i|z_i \geq 0)=x^T\beta+\frac{\phi(x^T\beta)}{\Phi(x^T\beta)}
$$

And If $y_i=0$,
$$
E(z_i|z_i<0) =x^T\beta-\frac{\phi(x^T\beta)}{\Phi(-x^T\beta)}
$$


{% highlight r %}
b <- c(10,2) #initial values for betas
ex <- cbind(1,x)
mu <- ex %*% b
it=1
converged = FALSE
maxits = 100000
tol=0.0001
gammas=array(0,20)

while ((!converged) & (it < maxits)) {
  b_old = b
  gammas = ifelse(y==1,mu+dnorm(mu)/pnorm(mu), mu-dnorm(mu)/pnorm(-mu)) #E-step
  b = solve(t(ex)%*%ex)%*%t(ex)%*%gammas   #Mstep
  mu = ex %*% b
  it = it + 1
  converged = max(abs(b_old - b)) <= tol
}
b
{% endhighlight %}



{% highlight text %}
##         [,1]
##   0.06238753
## x 0.21378264
{% endhighlight %}



{% highlight r %}
it
{% endhighlight %}



{% highlight text %}
## [1] 19
{% endhighlight %}

Note that the estimations are not at all sensitive to initial values.
