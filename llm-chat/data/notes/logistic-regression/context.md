![[20221106_233612.jpg|500]]

Say you have given given mouse $i$ of age $a_i$ and weight $w_i$ a dose $d_i$ of your drug, and it has died ($y_i$=1) or not ($y_i=0$). Or for each $(a,w,d)$ combination you have measured 10 mice, of which a fraction $0\leq y_i\leq1$ has died.
Let's call $(x_1,x_2,x_3,x_4)=(1,a,w,d)=$.

Now you're interested in
- predicting the death chance given a new mouse's age, weight and dose.
- explaining what's going on - like analyzing if weight is relevant
- estimating overall death rates

Logistic regression takes the independent variables of each measurement and maps them onto a single scalar number $\lambda_i=\sum_j\alpha_jx_{i,j}$. This is then the argument to the logistic function (the **logit parameter**)
$\pi(\lambda)=\dfrac{1}{1+\exp(-\lambda)}\quad\Leftrightarrow\quad\lambda=\log(\dfrac{\pi}{1-\pi})$

Now $\pi(\vec{\alpha})$ is the modeled mean value for the given data values $y$. The $y$ values are assumed to follow some distro, usually a binomial, with mean $\pi$.
The product of these distros for all data entries yields a likelihood function.

MLE then yields the estimators $\hat{\alpha}_j$, the errors for which it can also deliver via the observed information matrix, GLM theory, or we just bootstrap it.

[[Generalized linear model]] statements apply: MLE minimizes deviance etc.

Problem: If training data is linearly separable and y values are 0s and 1s, MLE diverges.