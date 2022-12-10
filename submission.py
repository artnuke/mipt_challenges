import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`)
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : float
            single number with MSE value of linear model (X.dot(w)) on the selected dataset.

        Comment: If Y is two-dimentional, average the error over both dimentions.
        """

        return np.mean((X.dot(w) - Y)**2)

    @staticmethod
    def mae(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`)
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return: float
            single number with MAE value of linear model (X.dot(w)) on the selected dataset.

        Comment: If Y is two-dimentional, average the error over both dimentions.
        """

        return np.mean(np.abs(X.dot(w) - Y))

    @staticmethod
    def l2_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return: float
            single number with L2 norm of the weighs matrix

        Computes the L2 regularization term for the weights matrix w.
        """

        return np.sum(w**2)

    @staticmethod
    def l1_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : float
            single number with L1 norm of the weighs matrix

        Computes the L1 regularization term for the weights matrix w.
        """

        return np.sum(np.abs(w))

    @staticmethod
    def no_reg(w):
        '''
        Simply ignores the regularization
        '''
        return np.zeros_like(w)

    @staticmethod
    def mse_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`)
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : numpy array of shape (`n_features`, `target_dimentionality`)

        Computes the MSE derivative for linear regression w.r.t w weights matrix.
        """

        return 2 * np.dot(X.T, np.dot(X, w) - Y) / np.prod(Y.shape)

    @staticmethod
    def mae_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`)
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : numpy array of shape (`n_features`, `target_dimentionality`)

        Computes the MAE derivative for linear regression w.r.t w weights matrix.
        """

        return np.dot(X.T, np.sign(np.dot(X, w) - Y)) / np.prod(Y.shape)

    @staticmethod
    def l2_reg_derivative(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : numpy array of shape (`n_features`, `target_dimentionality`)

        Computes the L2 regularization term derivative w.r.t the weights matrix w.
        """

        return 2. * w

    @staticmethod
    def l1_reg_derivative(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)

        Return : numpy array of shape (`n_features`, `target_dimentionality`)

        Computes the L1 regularization term derivative w.r.t the weights matrix w.
        """

        return np.sign(w).astype(float)

    @staticmethod
    def no_reg_derivative(w):
        '''
        Simply ignores the derivative
        '''
        return np.zeros_like(w)
