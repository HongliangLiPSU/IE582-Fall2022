# IE582 Fall 2022 Assignment-1

**Hongliang Li, hjl5377@psu.edu, 10/01/2022**

**Q1 Solution**

* Compute estimates of the slope $\beta_1$ and the intercept $\beta_0$ for the simple linear regression.

  According to the question, the matrix form of the independent variable and dependent variable are:

  $$
  \begin{align}
  X&=
  \begin{bmatrix}
  1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
  169.6 & 166.8 & 157.1 & 181.1 & 158.4 & 165.6 & 166.7 & 156.5 & 168.1 & 165.3\\
  \end{bmatrix}^{\mathrm{T}}\\
  Y&= 
  \begin{bmatrix}
  71.2 & 58.2 & 56.0 & 64.5 & 53.0 & 52.4  & 56.8  & 49.2 & 55.6 & 77.8\\
  \end{bmatrix}^{\mathrm{T}}
  \end{align}
  $$

  $\hat{\beta}$ can be estimated as:

  $$
  \hat{\beta} = (X^{\mathrm{T}}X)^{-1}X^{\mathrm{T}}Y
  $$
  
  It can be calculated as:
  
  $$
  \begin{align}
  (X^{\mathrm{T}}X)^{-1} &=
  \begin{bmatrix}
  58.13 & -0.3506 \\
  -0.3506 & 0.0021 \\
  \end{bmatrix}\\
  X^{\mathrm{T}}Y &= 
  \begin{bmatrix}
  594.7 \\
  98709.53\\
  \end{bmatrix}\\
  \Rightarrow \hat{\beta} &= 
  \begin{bmatrix}
  -36.88 \\
  0.58\\
  \end{bmatrix}
  \end{align}
  $$
  
  Hence, $\beta_{0}=-36.88$, $\beta_{1}=0.58$

* Find the residuals associated with the fitted model. Calculate associated mean and standard deviation(sample).

  Residual can be calculated as:
  
  $$
  \begin{align}
  r &= y - \hat{y} \\
  \Rightarrow r &=\begin{bmatrix}
  71.2 & 58.2 & 56.0 & 64.5 & 53.0 & 52.4  & 56.8  & 49.2 & 55.6 & 77.8\\
  \end{bmatrix}^{\mathrm{T}}\\
  &-\begin{bmatrix}
  61.8 & 60.2 & 54.6 & 68.5 & 55.3 & 59.5 & 60.2 & 54.2 & 61.0 & 59.3\\
  \end{bmatrix}^{\mathrm{T}}\\
  &= \begin{bmatrix}
  9.4 & -2.0 & 1.4 & -4.0 & -2.3 & -7.1 & -3.4 & -5.0 & -5.4 & 18.5\\
  \end{bmatrix}^{\mathrm{T}}
  \end{align}
  $$
  
  The associated mean and standard deviation can be calculated as:
  
  $$
  \begin{align}
  \bar{x}_r &= \sum_{i=1}^{10}r_i=0 \\
  \sigma_r &= \sqrt{\sum_{i=1}^n(r_i-\bar{x}_r)^2 \over (N-1)}=7.97
  \end{align}
  $$
  
* Estimate $R^2$ for the fitted line. What would be the correlation between two variables?

  SST and SSE can be calculated as:
  
  $$
  \begin{align}
  SST &= \sum_{i=11}^N(y_i-\bar{y})^2=731.96 \\
  SSE &= \sum_{i=1}^Nr_i^2=572.01 \\
  \Rightarrow R^2 &= {{SST-SSE}\over{SST}}=0.2185
  \end{align}
  $$
  
  The independent variable has the positive effect on the dependent variable.
  
* Predict the weight range for a group having the height range in 140 cm to 190 cm.
  
  According to the above calculations, we have:
  
  $$
  \begin{align}
  \hat{y_i}&=-36.88+0.85x_i \\
  \Rightarrow x_1&=140, \hat{y_1}=44.6 \\
  x_2&=190, \hat{y_2}=73.7\\
  \end{align}
  $$
  
  Hence, $x_i\in[140,190]$, $\hat{y_i}\in[44.6,73.7]$.

**Q2 Solution**

* Based on the provided set of equations, state the necessary assumptions so that the problem can be converted into **Multiple Linear Regression**.
  
  Assumptions:
  
  1. Assume the linearity between y and x as $E(Y│X)=\beta_\theta+∑_{j=1}^p{X_j\beta_j}$
  
  2. $\varepsilon$ is normal distributed as $Y=E(Y│X)+\varepsilon$. $X_{ij}$ is fixed and $Y_i$ is random due to the $\varepsilon$.
  
  3. $\varepsilon$ is independent of X.
  
  4. $X_j$ are linearly independent so as to calculate $(X^{\mathrm{T}}X)^{-1}$.
  
  5. The errors, $\varepsilon_i$, at each set of values of the predictors have equal variances.

* We discussed in the class that the closed-form solution for above problem by using **Ordinary Least Square (OLS)** is 
  
  $$
  \boldsymbol{\beta}_{ols}^*=(\boldsymbol{X}^\top\boldsymbol{X})^{-1}\boldsymbol{X}^\top\boldsymbol{y}
  $$
  where 
  $$
  \boldsymbol{X}=\left(\begin{array}{ccccc}
  1 & x_{1,1} & x_{1,2} & \cdots & x_{1, d} \\
  1 & x_{2,1} & x_{2,2} & \cdots & x_{2, d} \\
  1 & \vdots & \vdots & \vdots & \vdots \\
  1 &x_{n, 1} & x_{n, 2} & \cdots & x_{n, d}
  \end{array}\right)
  $$
  $$
  \boldsymbol{y} = \left(\begin{array}{c}
  y_1 \\
  y_2 \\
  \vdots \\
  y_n
  \end{array}\right)
  $$
 
  Verify this solution with the simple linear regression equations.
 
  For the simple linear regression, we have
 
  $$
  Y=X\beta+\varepsilon
  $$
 
  Based on the OLS to minimize the $\varepsilon^{\mathrm{T}}\varepsilon$, we have
 
  $$
  \varepsilon^{\mathrm{T}}\varepsilon=(Y-X\beta)^{\mathrm{T}}(Y-X\beta)
  $$
 
  Take the derivative, we have
 
  $$
  {d\over d\beta}(Y-X\beta)^{\mathrm{T}}(Y-X\beta)=-2X^{\mathrm{T}}(Y-X\beta)
  $$
 
  In order to minimize the $\varepsilon^{\mathrm{T}}\varepsilon$, set the derivative result equal to 0, then we have
 
  $$
  \begin{align}
  -2X^{\mathrm{T}}(Y-X\beta)&=0 \\
  \Rightarrow X^{\mathrm{T}}(Y-X\beta)&=0 \\
  \Rightarrow X^{\mathrm{T}}Y&=X^{\mathrm{T}}X\beta\\
  \Rightarrow \beta &= (X^{\mathrm{T}}X)^{-1}X^{\mathrm{T}}Y
  \end{align}
  $$
 
  Q.E.D
 
* We say **OLS** is not a stabilized solution for the Multiple Linear Regression problem. Please explain why?
  
  $X_j$ are linearly independent so as to calculate $X^{\mathrm{T}}X$. This assumption may not be true.
  
* Ridge Regression is a solution for the issues of OLS by adding the $\ell_2$ norm for the parameter vector as regularizer on the objective function:
  $$
    f_{RR}(\boldsymbol{\beta}) = \frac{1}{2}\|\boldsymbol{X}\boldsymbol{\beta}-\boldsymbol{y}\|^2_2+\lambda\|\boldsymbol{\beta}\|_2^2
  $$
  where $\lambda$ denotes the regularization parameter.
  Find the optimal solution with respect to $\boldsymbol{\beta}$ in matrix notation. Please provide sufficient reasoning for your steps.
  
  Let's consider a simple linear regression problem
  
  $$
  y=X\beta+\varepsilon
  $$
  
  Estimate $\beta$ by OLS with $l_2$ norm, we have
  
  $$
  \underset {\theta}{min}\left \| y-X\beta \right \|^2+\lambda\left \| \beta \right \|^2
  $$
  
  In order to solve this, take the derivative then we have
  
  $$
  {d\over d\beta}=2X^{\mathrm{T}}(X\beta-y)+2\lambda\beta
  $$
  
  Set the derivative equal to 0, we have
  
  $$
  \begin{align}
  {d\over d\beta}=2X^{\mathrm{T}}(X\beta-y)+2\lambda\beta &= 0\\
  \Rightarrow X^{\mathrm{T}}(X\beta-y)+\lambda\beta &= 0 \\
  \Rightarrow X^{\mathrm{T}}X\beta-X^{\mathrm{T}}y+\lambda\beta &= 0\\
  \Rightarrow (X^{\mathrm{T}}X+\lambda I)\beta &= X^{\mathrm{T}}y \\
  \Rightarrow \beta &= (X^{\mathrm{T}}X+\lambda I)^{-1}X^{\mathrm{T}}y
  \end{align}
  $$
  
  Q.E.D

**Q3 Solution**

* Your final estimators of the model. Indentify factors that are negatively related to count of rental bikes.
  
  The calculated estimators are all positive. It indicates that there is no negative relation between the dependent variable and one of the independent variables.

* Plot the distribution of residuals of your model. Discuss the residuals of your model based on your Multiple Linear Regression assumptions.
  
  ![avatar](https://raw.githubusercontent.com/HongliangLiPSU/IE582-Fall2022/main/HW1/LearningCurve.png)
  
* Try different learning rate and plot the learning curve in the same plot.
  
  ![avatar](https://raw.githubusercontent.com/HongliangLiPSU/IE582-Fall2022/main/HW1/DiffLearningRate.png)

* List possible two actions that can be taken to improve the model performance. (You are only required to list actions.)
  
  1. First, adjusting a learning rate, which is alpha here.
  2. Second, the collinearity occurs among the different predictor variables which violates the assumption of linear regressions. The possible methods to solve this problem is using Ridge regression.
 
 
  
  