# ==========================
# Program 7
# ==========================
# Question:
# You are comparing the average daily sales between two stores. Store A has a mean daily sales value of $1,000 with a standard deviation of $100 over 30 days,
# and Store B has a mean daily sales value of $950 with a standard deviation of $120 over 30 days.
# Conduct a two-sample t-test to determine if there is a significant difference between the average sales of the two stores at the 5% significance level.

mean_a = 1000
std_a = 100
n_a = 30
mean_b = 950
std_b = 120
n_b = 30
mean_diff = mean_a - mean_b
se = ((std_a ** 2) / n_a + (std_b ** 2) / n_b) ** 0.5
t_stat = mean_diff / se
df_numerator = ((std_a ** 2) / n_a + (std_b ** 2) / n_b) ** 2
df_denominator = (((std_a ** 2) / n_a) ** 2) / (n_a - 1) + (((std_b ** 2) / n_b) ** 2) / (n_b - 1)
df = df_numerator / df_denominator
print("\nProgram 7 Output:")
print(f"T-statistic: {t_stat:.4f}")
print(f"Approximate Degrees of Freedom: {df:.2f}")
if abs(t_stat) > 2.004:
    print("Result: Reject H₀ → Significant difference in sales.")
else:
    print("Result: Fail to reject H₀ → No significant difference in sales.")


# ==========================
# Program 8 (Part 1)
# ==========================
# Question:
# A company collects data on employees’ salaries and their education level (High School, Bachelor’s, Master’s).
# Fit a multiple linear regression model to predict salary using education level and years of experience.
# Interpret coefficients for education levels in the regression model.

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.DataFrame({
    'Salary': [40000, 50000, 60000, 70000, 80000, 55000, 65000, 75000, 85000],
    'Education': ['High School', 'Bachelor\'s', 'Master\'s', 'High School', 'Bachelor\'s', 'Master\'s', 'High School', 'Bachelor\'s', 'Master\'s'],
    'Experience': [2, 3, 4, 5, 6, 7, 3, 5, 8]
})
data['Education'] = pd.Categorical(data['Education'], categories=['High School', 'Bachelor\'s', 'Master\'s'])
model = smf.ols('Salary ~ C(Education) + Experience', data=data).fit()
print("\nProgram 8 (Part 1) Output:")
print(model.summary())


# ==========================
# Program 8 (Part 2)
# ==========================
data = [
    [40000, "High School", 2],
    [50000, "Bachelor's", 3],
    [60000, "Master's", 4],
    [70000, "High School", 5],
    [80000, "Bachelor's", 6],
    [55000, "Master's", 7],
    [65000, "High School", 3],
    [75000, "Bachelor's", 5],
    [85000, "Master's", 8],
]
X = []
y = []
for row in data:
    salary, education, experience = row
    bachelors = 1 if education == "Bachelor's" else 0
    masters = 1 if education == "Master's" else 0
    X.append([1, bachelors, masters, experience])
    y.append(salary)
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]
def matmul(A, B):
    return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*B)] for row_a in A]
def inverse_matrix_4x4(A):
    import numpy as np
    return np.linalg.inv(A).tolist()
X_T = transpose(X)
XTX = matmul(X_T, X)
XTY = matmul(X_T, [[val] for val in y])
inv_XTX = inverse_matrix_4x4(XTX)
beta = matmul(inv_XTX, XTY)
print("\nProgram 8 (Part 2) Output:")
coeff_names = ["Intercept", "Bachelor's", "Master's", "Experience"]
for name, coef in zip(coeff_names, beta):
    print(f"{name}: {coef[0]:.2f}")


# ==========================
# Program 9 (Part 1)
# ==========================
# Question:
# You have data on housing prices and square footage.
# Fit a spline regression model to allow the relationship between square footage and price to change at 2,000 sqft.

data = [
    [200000, 1500],
    [220000, 1700],
    [250000, 2000],
    [270000, 2100],
    [290000, 2300],
    [320000, 2500],
]
X = []
y = []
for price, sqft in data:
    x1 = sqft
    x2 = max(0, sqft - 2000)
    X.append([1, x1, x2])
    y.append(price)
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]
def matmul(A, B):
    return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*B)] for row_a in A]
def inverse_matrix_3x3(A):
    import numpy as np
    return np.linalg.inv(A).tolist()
X_T = transpose(X)
XTX = matmul(X_T, X)
XTY = matmul(X_T, [[val] for val in y])
beta = matmul(inverse_matrix_3x3(XTX), XTY)
print("\nProgram 9 (Part 1) Output:")
print(f"Intercept: {beta[0][0]:.2f}")
print(f"Before 2000 sqft slope: {beta[1][0]:.2f}")
print(f"Change after 2000 sqft: {beta[2][0]:.2f}")


# ==========================
# Program 9 (Part 2)
# ==========================
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = {'sqft': [1500, 1700, 1900, 2000, 2100, 2300, 2500],
        'price': [200000, 220000, 240000, 260000, 290000, 320000, 350000]}
df = pd.DataFrame(data)
df['spline'] = np.where(df['sqft'] > 2000, df['sqft'] - 2000, 0)
X = sm.add_constant(df[['sqft', 'spline']])
y = df['price']
model = sm.OLS(y, X).fit()
print("\nProgram 9 (Part 2) Output:")
print(model.summary())


# ==========================
# Program 10
# ==========================
# Question:
# A hospital uses a Poisson regression model to predict the number of ER visits per week based on Age and Condition.
# log(λ) = 2.5 - 0.03 * Age + 0.5 * Condition

intercept = 2.5
beta_age = -0.03
beta_condition = 0.5
data = pd.DataFrame({'Age': [60, 60], 'Condition': [1, 0]})
data['log_lambda'] = intercept + beta_age * data['Age'] + beta_condition * data['Condition']
data['lambda'] = np.exp(data['log_lambda'])
increase_pct = ((data.loc[0, 'lambda'] - data.loc[1, 'lambda']) / data.loc[1, 'lambda']) * 100
print("\nProgram 10 Output:")
print(data[['Age', 'Condition', 'lambda']])
print(f"\nIncrease in expected visits due to chronic condition: {increase_pct:.2f}%")


# ==========================
# Program 11
# ==========================
# Question:
# A bakery claims its new cookie recipe is lower in calories than the old recipe (mean = 200).
# A sample of 40 cookies shows mean = 190 and SD = 15. Perform a one-tailed t-test at α = 0.05.

import math
from scipy import stats

mu_0 = 200
x_bar = 190
s = 15
n = 40
alpha = 0.05
t_stat = (x_bar - mu_0) / (s / math.sqrt(n))
df = n - 1
t_critical = stats.t.ppf(alpha, df)
p_value = stats.t.cdf(t_stat, df)
print("\nProgram 11 Output:")
print(f"T-statistic: {t_stat:.3f}")
print(f"Critical t-value: {t_critical:.3f}")
print(f"P-value: {p_value:.5f}")
if t_stat < t_critical:
    print("Reject the null hypothesis: The new recipe has significantly fewer calories.")
else:
    print("Fail to reject the null hypothesis: Not enough evidence to support the claim.")
