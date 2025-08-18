"Learn a function for which we have noisy observations"

model:
$\vec{y}=\boldsymbol{X}\vec{\beta}+\vec{\epsilon}$
where
==y==:data points
==X==: nxp structure matrix, X=(1,1,1;x_1,x_2,x_3,...) but **the entries of X are not necessarily linear in the independent variables x**: you can pre-transform your data, and use say sin(x) as a row in X!
Really, linear regression can therefor be understood as a linear superposition of base functions.
==beta==: unknown p-dim- parameter vector
==epsilon==: noise ~ N(0,sigma^2)

MLE minimizes least squares:
$\hat{\beta}=\arg\min ||\vec{y}-\boldsymbol{X}\vec{\beta}||^2$

and has analytic solution:
$\hat{\beta}=\boldsymbol{S}^{-1}\boldsymbol{X}'\vec{y}$

where $\boldsymbol{S}=\boldsymbol{X}'\boldsymbol{X}$ is pxp.

extra:
to obtain an unbiased estimate for noise variance (around regression line) the mean squared error has to be modified b.c. data used to estimate noise variance was also used for finding regression line:
$\sum_i(y_i-\hat{y}_i)/(N-p)$
this is similar to unbiased std estimator dividing by N-1