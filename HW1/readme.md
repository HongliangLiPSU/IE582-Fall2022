
# IE582 Fall 2022 Assignment-1
**Due Date - 10/02/2022**

**Instructions**

* Please complete this assignment in the group you formed.
* You are given three question sets.
    * Question 1: Simple Linear Regression Calculation
    * Question 2: Multiple Linear Regression Matrix Derivation
    * Question 3: Gradient Descent for Linear Regression
* Programming is not required to solve Question 1 and Question 2 (You can use programming if you think it helps you solve the problem). For Question 3, both **Python** and **R** are accepted. But the starter code for Python is given.
* We are aware that there are similar questions and solutions available online. If you leverage anyone else's work, please cite the source and highlight where/how you applied it to your own solution.
* Please submit the solution in a single `.pdf` ir `.md` file.

**Question 1**
You are given a dataset that records the `ht` (Height in cms) and `wt` (Weight in kgs) for `n=10` 18-year-old girls. Assuming `ht` is **the independent variable**, please answer the following questions based on this dataset.

<center>

| |ht |wt| 
|:-:|:-----:|:------:|
|1|169.6|71.2|
|2|166.8|58.2|
|3|157.1|56.0|
|4|181.1|64.5|
|5|158.4|53.0|
|6|165.6|52.4|
|7|166.7|56.8|
|8|156.5|49.2|
|9|168.1|55.6|
|10|165.3|77.8|

</center>

* Compute estimates of the slope $\beta_1$ and the intercept $\beta_0$ for the simple linear regression.
* Find the residuals associated with the fitted model. Calculate associated mean and standard deviation(sample).
* Estimate $R^2$ for the fitted line. What would be the correlation between two variables?
* Predict the weight range for a group having the height range in 140 cm to 190 cm.

**Question 2**

Consider the following equation for $i^{th}$ observation of $y$ dependent variable using $X$ as an independent variable.
$$
y_i = \beta_0 + \sum_{j=1}^{p}\beta_{j}x_{ij} + e_i
$$
Where
$$
i = i,2,\dots, n
$$


* Based on the provided set of equations, state the necessary assumptions so that the problem can be converted into **Multiple Linear Regression**.
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
* We say **OLS** is not a stabilized solution for the Multiple Linear Regression problem. Please explain why?
* Ridge Regression is a solution for the issues of OLS by adding the $\ell_2$ norm for the parameter vector as regularizer on the objective function:
  $$
    f_{RR}(\boldsymbol{\beta}) = \frac{1}{2}\|\boldsymbol{X}\boldsymbol{\beta}-\boldsymbol{y}\|^2_2+\lambda\|\boldsymbol{\beta}\|_2^2
  $$
  where $\lambda$ denotes the regularization parameter.
  Find the optimal solution with respect to $\boldsymbol{\beta}$ in matrix notation. Please provide sufficient reasoning for your steps.


**Question 3:**

This assignment is based on the [bike sharing dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset#) on UCI Machine Learning Repository. You are asked to build a Multiple Linear Regression Model on the **daily count of rental bikes**. Try to give the optimized estimator by using the **OLS**. Note that since taking inverse in closed-form solution of OLS is expensive (the time complexity is $O(d^3)$), you are required to use **Gradient Descent (GD)** to solve the optimization problem. 

**Deliverables:**

1. Your final estimators of the model. Indentify factors that are negatively related to count of rental bikes.
2. Plot the distribution of residuals of your model. Discuss the residuals of your model based on your Multiple Linear Regression assumptions.
3. Try different learning rate and plot the learning curve in the same plot.
4. List possible two actions that can be taken to improve the model performance. (You are only required to list actions.)


**Tips:**
* Please carefully read the dataset information provided by the [UCI website](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset#).
* You don't need to use `dteday`, `casual`, `registered` variables in this assignment. 
* Check if there are any missing data. If so, fill them with `0`.
* The starter code has been provided.
* You don't need any external machine learning packages to solove the problem.