# Bayesian Optimizer with Gaussian Process Regression
 Bayesian Optimization with Gaussian Process Regression for Function Approximation

## Theory

### Gaussian Process Regression (GPR)
Gaussian Process Regression is a powerful non-parametric regression technique that models the target function as a Gaussian process. It provides a probabilistic framework to make predictions and estimate uncertainties; where, the training data is used to learn the underlying distribution of the target function, and predictions are made based on the learned distribution.

At its core, a Gaussian process defines a probability distribution over functions. Instead of predicting a single value for a given input, a GP provides a distribution of possible functions that could have generated the data. This distribution is typically represented by a mean function $\mu$ and a covariance function or kernel $k$.

```math
f \sim GP(\mu(\bar{x}); k(\bar{x}, \bar{x}'))
```

The mean function represents the expected value of the function at each input point. It provides a baseline estimate of the function's behavior. The covariance function  characterizes the similarity between input points. It determines how the function values at different input points are related to each other. Moreover, the choise of the covariance function drives the differentiability of the target function and should allow a good definition of the covariance matrix (symmetric and positive definite). The covariance function essentially encodes our prior beliefs about the smoothness and behavior of the underlying function.

Given some observed data points, a GP can be used to make predictions at new input points by conditioning the distribution on the observed data. The GP uses the observed data to update its prior beliefs and provides a posterior distribution over functions. This posterior distribution captures the uncertainty associated with the predictions, taking into account the noise or measurement errors in the data. 

### Bayesian Optimization
Bayesian Optimization is an optimization technique that utilizes probabilistic models to sequentially evaluate points in the search space. It balances exploration (finding new promising points) and exploitation (evaluating points likely to yield better results) to efficiently optimize the target function. Bayesian Optimization uses an acquisition function to select the next point for evaluation based on the trade-off between exploration and exploitation.

The main steps of Bayesian Optimization are as follows:
1. Initialize the training data with a few initial points.
2. Fit a Gaussian Process Regression model to the training data.
3. Select the next point to evaluate using an acquisition function.
4. Evaluate the selected point and add it to the training data.
5. Update the Gaussian Process Regression model with the new data.
6. Repeat steps 3-5 until a termination criterion is met.

## Code Structure

This repository contains code that implements Gaussian Process Regression (GPR) and Bayesian optimization for function approximation. It consists of two main classes:

- GPR: Represents the Gaussian Process Regression model. It provides methods to fit the model to training data, predict the mean and covariance of the target function, and define the kernel function for GPR. The fit method performs hyperparameter optimization using negative log-likelihood loss, and the predict method predicts the mean and covariance of the target function based on the fitted model.

- BayesianOptimization: Performs the Bayesian optimization process. It takes a target function, a domain range, initial points, and the number of optimization steps as inputs. In the constructor, it initializes the GPR model, fits it to the initial training data, and visualizes the initial predictions. The get_next_guess method calculates the next point to evaluate based on expected improvement, and the maximise method performs the iterative optimization process. It selects the next point, appends it to the training data, fits the GPR model, and visualizes the updated predictions at each step.

## Dependencies
The code requires the following dependencies:

- scipy.optimize
- numpy
- matplotlib.pyplot
- scipy.stats

## Usage
To use the code, follow these steps:

1. Define the target function for Bayesian Optimization. The function should take an input x and return a corresponding output y.

2. Create an instance of the BayesianOptimization class, providing the target function, domain range (minimum and maximum values), initial points, and the number of optimization steps.

3. Call the `maximise()` or `best_fit()` method of the BayesianOptimization instance to perform the Bayesian optimization process and find the maximum or best fit of the target function, respectively.

```python
# Define the target function for Bayesian Optimization
def y(x):
    x = np.asarray(x)
    y = np.sin(x) + np.sin((10.0 / 3.0) * x)
    return y.tolist()

# Create an instance of BayesianOptimization and perform maximization or best fit
bayes_opt = BayesianOptimization(target_function=y, domain_min=-4, domain_max=16, initial_points=[3, 4, 5, 9])
# bayes_opt.maximise()  # Perform maximization
bayes_opt.best_fit()  # Perform best fit
```
**Note:** The plots are saved in the output directory.
 
## Results
The code will fit the GPR model to the initial training data and visualize the initial predictions. Then, it will perform the iterative Bayesian optimization process and update the predictions at each step. The resulting plots will show the true function, the predicted function, the training data points, and the uncertainty region.

Please note that you can choose to perform either maximization or best fit by uncommenting the corresponding method call in the main code section.

## Conclusion
Bayesian Optimization with Gaussian Process Regression is a powerful technique for optimizing unknown functions that are expensive to evaluate. By utilizing a probabilistic model and intelligent point selection, it efficiently explores the search space and finds the optimum with fewer evaluations compared to traditional optimization methods. This code provides a simple implementation of Bayesian Optimization using Gaussian Process Regression and can serve as a starting point for more complex optimization tasks. 