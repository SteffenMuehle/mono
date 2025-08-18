6min intro: https://www.youtube.com/watch?v=FD4DeN81ODY

We have a large number of data points, possibly each being high-dimensional.
Let us identify directions $\vec{u] in that high-dim. space onto whcih to project them and maintain as much variance of the point cloud as possible:

max $\dfrac{1}{n}\sum_{i}(\vec{x}_{i}^{T}\vec{u})^2$ with the constraint that $\vec{u}^T\vec{u}=1$

some matrix algebra:
$\dfrac{1}{n}\sum_{i}(\vec{x}_{i}^{T}\vec{u})^2=\dfrac{1}{n}\sum_{i}\vec{u}^T\vec{x}_i\vec{x}_{i}^{T}\vec{u}=\vec{u}^TC\vec{u}$

where $C$ is the covariance matrix.
Take constraint into account via Lagrangian multiplier:
max $\vec{u}^TC\vec{u} - \lambda(\vec{u}^T\vec{u}-1)$
gradient w.r.t. $\vec{u}$:
$0\stackrel{!}{=}2C\vec{u}-2\lambda\vec{u}$
gives the eigenvalue equation:
$C\vec{u}=\lambda\vec{u}$