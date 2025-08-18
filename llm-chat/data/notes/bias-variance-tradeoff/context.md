a good model has it both: an accurate prediction (low bias), while not being overly sensitive to fluctuations in the training data (low variance).
Minimizing bias/variance alone generally leads to bad performance of the other, as there is a tradeoff:

==quantitative:==
From a true unknown pdf $p(\vec{x},y)$ we have collected a sample of $x,y$ pairs. The true $E(y|x)$ follows an unknown function $f(x)$. From the sample we fit a model which is our estimator for $f$: $\hat{f}(\vec{x}_0)$ for new data points $x_0$'s true $y$ value $f(x_0)$. 
squared prediction error: $\langle (f(x)-\hat{f}(x))^2\rangle$ follows the ==bias-variance decomposition==: 
prediction error = bias + variance + irreducible error
this error has the underlying **paradigm of repeatedly receiving different sample sets on which to base our fit and prediction**

==Extremes==
- high bias, low variance: "underfitting", "dumb model", few parameters. Extreme example: model always predicts 1, no matter what the training data. **Stubborn, assumption-driven models**. If assumptions are wrong, they are wrong even for many (or large) training data sets.
- low bias, high variance: "overfitting", overinterpreting the training data, too many parameters. Extreme example: n-parameter polynomial fit to n data points -> training data reproduced exactly, but prediction is VERY sensitive to training data. If you could get a LOT of training sets, we could caputre the true value really well, whatever it is! But on a SINGLE training set, the variance is high!
- ridge regression and the lasso are DELIBERATE ways to lower variance, paying the price by accepting some bias towards lower slopes

==Hyperparameters== a.k.a. top-level parameters such as
- number of layers in neural network
- degree of fitted polynomial
- ridge/lasso regression regularization parameter $\lambda$
- kernel width
- maximum tree depth in random forest
.. quantify out choice on the bias-variance tradeoff