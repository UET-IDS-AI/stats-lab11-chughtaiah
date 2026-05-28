import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.

    Return:
        X, y, true_coef
    """

    X, y, coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )

    return X, y, float(coef)


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers to the first n_outliers observations.

    Important:
    Do not modify original X and y directly.
    """

    rng = np.random.RandomState(random_state)

    X_out = X.copy()
    y_out = y.copy()

    X_out[:n_outliers] = 10 + 0.75 * rng.normal(size=(n_outliers, 1))
    y_out[:n_outliers] = -15 + 20 * rng.normal(size=n_outliers)

    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot dataset and highlight artificial outliers.

    Return:
        matplotlib Figure object
    """

    fig, ax = plt.subplots()

    ax.scatter(
        X[n_outliers:, 0],
        y[n_outliers:],
        label="Normal data",
        alpha=0.7
    )

    ax.scatter(
        X[:n_outliers, 0],
        y[:n_outliers],
        label="Artificial outliers",
        marker="x",
        s=70
    )

    ax.set_title("Dataset with Artificial Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    """
    Fit ordinary Linear Regression.

    Return:
        fitted coefficient as a float
    """

    model = LinearRegression()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_huber_regression(X, y):
    """
    Fit Huber Regression.

    Return:
        fitted coefficient as a float
    """

    model = HuberRegressor()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    """
    Fit RANSAC Regression.

    Return:
        fitted coefficient as a float
    """

    model = RANSACRegressor(
        estimator=LinearRegression(),
        random_state=random_state
    )

    model.fit(X, y)

    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    """
    Fit Theil-Sen Regression.

    Return:
        fitted coefficient as a float
    """

    model = TheilSenRegressor(random_state=random_state)
    model.fit(X, y)

    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    """
    Given a dictionary of coefficients and the true coefficient,
    return a dictionary of absolute coefficient errors.
    """

    return {
        name: abs(float(coef) - float(true_coef))
        for name, coef in coef_dict.items()
    }


def best_robust_model(errors):
    """
    Return the robust model with the smallest error.

    Only compare:
        huber_regression
        ransac_regression
        theilsen_regression
    """

    robust_models = [
        "huber_regression",
        "ransac_regression",
        "theilsen_regression"
    ]

    return min(
        robust_models,
        key=lambda model_name: errors[model_name]
    )


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Fit RANSAC and return:

        total_outliers_detected, added_outliers_detected
    """

    model = RANSACRegressor(
        estimator=LinearRegression(),
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    total_outliers_detected = int(np.sum(outlier_mask))
    added_outliers_detected = int(np.sum(outlier_mask[:n_outliers]))

    return total_outliers_detected, added_outliers_detected


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot fitted regression lines for:
    - Linear Regression
    - Huber Regression
    - RANSAC Regression
    - Theil-Sen Regression

    Return:
        matplotlib Figure object
    """

    linear_model = LinearRegression()
    huber_model = HuberRegressor()
    ransac_model = RANSACRegressor(
        estimator=LinearRegression(),
        random_state=random_state
    )
    theilsen_model = TheilSenRegressor(random_state=random_state)

    linear_model.fit(X, y)
    huber_model.fit(X, y)
    ransac_model.fit(X, y)
    theilsen_model.fit(X, y)

    x_grid = np.linspace(
        X.min(),
        X.max(),
        300
    ).reshape(-1, 1)

    fig, ax = plt.subplots()

    ax.scatter(
        X[:, 0],
        y,
        alpha=0.5,
        label="Data"
    )

    ax.plot(
        x_grid[:, 0],
        linear_model.predict(x_grid),
        label="Linear Regression"
    )

    ax.plot(
        x_grid[:, 0],
        huber_model.predict(x_grid),
        label="Huber Regression"
    )

    ax.plot(
        x_grid[:, 0],
        ransac_model.predict(x_grid),
        label="RANSAC Regression"
    )

    ax.plot(
        x_grid[:, 0],
        theilsen_model.predict(x_grid),
        label="Theil-Sen Regression"
    )

    ax.set_title("Regression Fits with Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Fit RANSAC and visualize inliers vs outliers.

    Return:
        matplotlib Figure object
    """

    model = RANSACRegressor(
        estimator=LinearRegression(),
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots()

    ax.scatter(
        X[inlier_mask, 0],
        y[inlier_mask],
        label="RANSAC inliers",
        alpha=0.7
    )

    ax.scatter(
        X[outlier_mask, 0],
        y[outlier_mask],
        label="RANSAC outliers",
        marker="x",
        s=70
    )

    x_grid = np.linspace(
        X.min(),
        X.max(),
        300
    ).reshape(-1, 1)

    ax.plot(
        x_grid[:, 0],
        model.predict(x_grid),
        label="RANSAC fit"
    )

    ax.set_title("RANSAC Inliers and Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig
