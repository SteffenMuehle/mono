an iterative process to construct a good fit function. Fit function = sum of tree outputs.
Into each iteration we feed not the original data, but the previous iteration's residuals (X,r): Improving the model where it currently fails.
In each iteration, add found fit TIMES EPSILON to "master function".

variations:
- **gradient boosting**: GLM-like curve fitting. Gradient descent on loss function. Find trees that whose output is close to wanted gradient.
- **lasso+boosting**: consider each iteration's found tree as a "base fucntion" and let the lasso decide (on top of boosting) which ones to keep and which ones to get rid of.
- **adaboost**:
	- each tree is a "stump": node with two leaves, that's it.
	- order matters: nth tree's making is influenced by the way in which n-1th tree failed: weighted bootstrap sample creation
	- some trees have other weight in final forest decision than other trees