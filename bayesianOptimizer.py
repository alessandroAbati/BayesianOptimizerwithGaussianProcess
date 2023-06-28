#Bayesian Optimization with Gaussian Process Regression for Function Approximation

"""
The code implements Gaussian Process Regression (GPR) and Bayesian optimization for function approximation.
It consists of two main classes: 
-GPR for Gaussian Process Regression;
-BayesianOptimization for performing Bayesian optimization.

The GPR class represents the Gaussian Process Regression model. 
It has methods to fit the model to training data, predict the mean and covariance of the target function, 
and define the kernel function for GPR. 
The fit method performs hyperparameter optimization using negative log-likelihood loss. 
The predict method predicts the mean and covariance of the target function based on the fitted model.

The BayesianOptimization class performs the Bayesian optimization process. 
It takes a target function, a domain range, initial points, and the number of optimization steps as inputs. 
In the constructor, it initializes the GPR model, fits it to the initial training data, and visualizes the initial predictions. 
The get_next_guess method calculates the next point to evaluate based on expected improvement. 
The maximise method performs the iterative optimization process. 
It selects the next point, appends it to the training data, fits the GPR model, and visualizes the updated predictions at each step.

In the main code section, a target function y is defined, and an instance of BayesianOptimization is created.
The maximise method is called to perform the Bayesian optimization process and find the maximum of the target function.
The best_fit method is called to perform the Bayesian optimization process and find the best approximation of the target function.
"""

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GPR:
    def __init__(self, optimize=True):
        # Initialize the GPR model
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # Fit the GPR model to the training data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        
        def negative_log_likelihood_loss(params):
            # Define the negative log-likelihood loss function for hyperparameter optimization
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            return (0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)).flatten()
        
        if self.optimize:
            # Optimize the hyperparameters of the GPR model using negative log-likelihood loss
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                           bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]
        
        self.is_fit = True

    def predict(self, X):
        # Predict the mean and covariance of the target function using the GPR model
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return
        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov
    
    def kernel(self, x1, x2):
        # Define the kernel function for Gaussian Process Regression
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

class BayesianOptimization:
    def __init__(self, target_function, domain_min: int, domain_max: int, initial_points: list, n_steps=30):
        # Initialize the Bayesian Optimization class
        self.f = target_function
        self.n_steps = n_steps
        self.train_X = np.array(initial_points).reshape(-1, 1)
        self.train_y = self.f(self.train_X)
        self.test_X = np.arange(domain_min, domain_max, 0.01).reshape(-1, 1)  
        self.gpr = GPR()
        
        # Fit the GPR model to the initial training data and visualize the initial predictions
        self.gpr.fit(self.train_X, self.train_y)
        mu, cov = self.gpr.predict(self.test_X)
        test_y = mu.ravel()
        uncertainty = np.sqrt(np.diag(cov))
        plt.figure()
        plt.plot(self.test_X, [self.f(x) for x in self.test_X], c='darkorange', label='true')
        plt.title("Initialization:  l=%.2f sigma_f=%.2f" % (self.gpr.params["l"], self.gpr.params["sigma_f"]))
        plt.fill_between(self.test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.15, color='royalblue')        
        plt.plot(self.test_X, test_y, label="predict", c='royalblue')
        plt.scatter(self.train_X, self.train_y, label="train", c="darkred", marker="x")
        plt.legend()
        plt.show()

    def get_next_guess_ei(self, mu, cov):
        # Calculate the next point to evaluate based on expected improvement
        y_max = np.max(self.train_y)
        expected_improvement = []
        mu = mu.ravel()
        z = (mu - y_max) / np.sqrt(np.diag(cov))
        expected_improvement = (mu - y_max) * norm.cdf(z) + np.diag(cov) * norm.pdf(z)
        next_guess = np.argmax(expected_improvement)
        
        return next_guess, expected_improvement
    
    def get_next_guess_bf(self, cov):
        #Calculate the next point to evaluate based on biggest uncertainty
        next_guess = np.argmax(np.diag(cov))
        return next_guess
    
    def maximise(self):
        # Perform iterative Bayesian optimization to find the maximum of the model
        mu, cov = self.gpr.predict(self.test_X)
        for i in range(self.n_steps):
            next_guess, ei = self.get_next_guess_ei(mu=mu, cov=cov)
            self.train_X = np.append(self.train_X, self.test_X[next_guess]).reshape(-1, 1)
            self.train_y.append(self.f(self.train_X[-1]))
            print(f'Step n° {i+1} --- Next point to evaluate: {self.train_X[-1]}') 
            self.gpr.fit(self.train_X, self.train_y)
            mu, cov = self.gpr.predict(self.test_X)
            test_y = mu.ravel()
            uncertainty = np.sqrt(np.diag(cov))
            plt.figure()
            plt.plot(self.test_X, (i+1)*ei, c='green', label = f'{i+1}*EI')
            plt.plot(self.test_X, [self.f(x) for x in self.test_X], c='darkorange', label='true')
            plt.title("l=%.2f sigma_f=%.2f" % (self.gpr.params["l"], self.gpr.params["sigma_f"]))
            plt.fill_between(self.test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.15, color='royalblue')        
            plt.plot(self.test_X, test_y, label="predict", c='royalblue')
            plt.scatter(self.train_X, self.train_y, label="train", c="darkred", marker="x")
            plt.legend()
            plt.show()

    def best_fit(self):
        # Perform iterative Bayesian optimization to find the best fit of the model
        mu, cov = self.gpr.predict(self.test_X)
        for i in range(self.n_steps):
            next_guess = self.get_next_guess_bf(cov=cov)
            self.train_X = np.append(self.train_X, self.test_X[next_guess]).reshape(-1, 1)
            self.train_y.append(self.f(self.train_X[-1]))
            print(f'Step n° {i+1} --- Next point to evaluate: {self.train_X[-1]}') 
            self.gpr.fit(self.train_X, self.train_y)
            mu, cov = self.gpr.predict(self.test_X)
            test_y = mu.ravel()
            uncertainty = np.sqrt(np.diag(cov))
            plt.figure()
            plt.plot(self.test_X, [self.f(x) for x in self.test_X], c='darkorange', label='true')
            plt.title("l=%.2f sigma_f=%.2f" % (self.gpr.params["l"], self.gpr.params["sigma_f"]))
            plt.fill_between(self.test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.15, color='royalblue')        
            plt.plot(self.test_X, test_y, label="predict", c='royalblue')
            plt.scatter(self.train_X, self.train_y, label="train", c="darkred", marker="x")
            plt.legend()
            plt.show()

if __name__ == '__main__':
    # Define the target function for Bayesian Optimization
    def y(x):
        x = np.asarray(x)
        y = np.sin(x) + np.sin((10.0 / 3.0) * x)
        return y.tolist()

    # Create an instance of BayesianOptimization and perform maximization
    bayes_opt = BayesianOptimization(target_function=y, domain_min=-4, domain_max=16, initial_points=[3, 4, 5, 9])
    #bayes_opt.maximise()
    bayes_opt.best_fit()
