![[ac4ckpii.bmp]]

Nonlinear curve fitting in a specific manner.
Ingredients:
1. A link function: The inverse of the curve to be fitted, i.e. x(y)
2. A probability distro describing the likelihood of data points with mean y(x)
3. linear predictor parametrizing x lienarly in terms of all independent variables.

This is a generalization of [[Logistic regression]]

==Advanced==:
- distro must be from [[Exponential family]] for this procedure to be called GLM. This brings nice properties with it.
- MLE minimizes deviance.
- If link function is the natural link function for the used distro, cool sufficient statistics stuff applies for repeated sampling or so.
- Hoeffdinger's Lemma tells us how quickly likelihood decreases around max value: depends only on deviance.
- https://stats.stackexchange.com/questions/439546/modeling-natural-parameters-in-glm
- https://towardsdatascience.com/generalized-linear-models-9cbf848bb8ab