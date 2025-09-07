### Program 1: IQR and Price Variability
import numpy as np
house_prices = [45, 55, 60, 62, 68, 70, 75, 80, 85, 90, 100, 110, 125]
q1 = np.percentile(house_prices, 25)
q3 = np.percentile(house_prices, 75)
iqr = q3 - q1
print(f"25th Percentile (Q1): {q1}")
print(f"75th Percentile (Q3): {q3}")
print(f"Interquartile Range (IQR): {iqr}")

### Program 2: Customer Satisfaction vs. Repeat Purchase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = {
    'Satisfaction': ['Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'High', 'Medium', 'Low',
                     'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'High', 'Medium', 'Low', 'High'],
    'Repeat Purchase': ['No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes',
                        'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Satisfaction', hue='Repeat Purchase')
plt.title('Customer Satisfaction vs Repeat Purchase')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Customers')
plt.legend(title='Repeat Purchase')
plt.tight_layout()
plt.show()
cross_tab = pd.crosstab(df['Satisfaction'], df['Repeat Purchase'])
cross_tab.plot(kind='bar', stacked=True, color=['salmon', 'skyblue'], figsize=(8, 5))
plt.title('Stacked Bar Chart: Satisfaction vs Repeat Purchase')
plt.xlabel('Satisfaction Level')
plt.ylabel('Number of Customers')
plt.legend(title='Repeat Purchase')
plt.tight_layout()
plt.show()

### Program 3: Car Features Correlation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = {
    'EngineSize_L': [1.2, 1.6, 2.0, 2.5, 3.0, 1.8, 2.2, 3.5, 4.0, 2.8],
    'FuelEfficiency_MPG': [40, 35, 30, 28, 24, 33, 29, 20, 18, 26],
    'Price_USD': [18000, 20000, 24000, 28000, 35000, 22000, 26000, 40000, 45000, 33000]
}
df = pd.DataFrame(data)
sns.pairplot(df)
plt.suptitle("Pair Plot: Engine Size, Fuel Efficiency, Price", y=1.02)
plt.tight_layout()
plt.show()
corr_matrix = df.corr(numeric_only=True)
print("Correlation Matrix:\n", corr_matrix)
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Car Features')
plt.tight_layout()
plt.show()

### Program 4: Central Limit Theorem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)
population = np.random.exponential(scale=70000, size=10000)
sample_means = []
for _ in range(10):
    sample = np.random.choice(population, size=50, replace=True)
    sample_means.append(np.mean(sample))
plt.figure(figsize=(8, 5))
sns.histplot(sample_means, bins=10, kde=True, color='skyblue', edgecolor='black')
plt.title("Sampling Distribution of Mean Salaries (10 samples of size 50)")
plt.xlabel("Sample Mean Salary")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

### Program 5: Drug Heart Rate t-Test
import scipy.stats as stats
sample_mean = 8
sample_std = 2
n = 20
mu = 0
t_statistic = (sample_mean - mu) / (sample_std / (n**0.5))
df = n - 1
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=df))
print(f"T-statistic: {t_statistic:.4f}")
print(f"Degrees of freedom: {df}")
print(f'P-value: {p_value:.4f}')
if p_value < 0.05:
    print("Result: Reject the null hypothesis. The increase is statistically significant.")
else:
    print("Result: Fail to reject the null hypothesis. The increase is not statistically significant.")

### Program 6: A/B Testing
import statsmodels.api as sm
x1 = 120
n1 = 1000
x2 = 150
n2 = 1200
count = [x1, x2]
nobs = [n1, n2]
z_stat, p_value = sm.stats.proportions_ztest(count, nobs)
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Reject the null hypothesis. There is a statistically significant difference in conversion rates.")
else:
    print("Result: Fail to reject the null hypothesis. No significant difference in conversion rates.")
