import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv('Original data/Students Social Media Addiction.csv')

### Data Exploration and Visualization

# Display basic info
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())


# Visualizing all features
plt.figure(figsize=(20, 15))

for i, col in enumerate(data.columns[1:], 1):  
    plt.subplot(3, 4, i)
    sns.histplot(data[col], bins=20, kde=True)
    plt.title(f'{col} Distribution')

plt.tight_layout()
plt.savefig('features_distribution.png')
plt.show()

# Correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(),annot=True, cmap='Purples', linewidths=0.1)
fig=plt.gcf()
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# Pairplot for key numerical features

sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Feature Relationships with Addiction Score', y=1.02)
plt.savefig('feature_pairplot.png')
plt.show()


### Data Preprocessing

# Convert categorical variables to numerical
label_encoders = {}
categorical_cols = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Affects_Academic_Performance', 'Relationship_Status']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
X = data.drop(['Student_ID', 'Addicted_Score', 'Country'], axis=1)  # Drop non-predictive columns
y = data['Addicted_Score']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Save preprocessed data 
X_train.to_csv('Preprocessed data\X.csv', index=False)
X_test.to_csv('Preprocessed data\X_test.csv', index=False)
y_train.to_csv('Preprocessed data\Y.csv', index=False, header=True)
y_test.to_csv('Preprocessed data\Y_test.csv', index=False, header=True)

### Model Training and Evaluation

models = {
    'LR': LinearRegression(),
    'SVM': SVR(),
    'RF': RandomForestRegressor(random_state=42),
    'DT': DecisionTreeRegressor(random_state=42),    
    'KNN': KNeighborsRegressor(),
    'NB': GaussianNB(),        
    'ANN': MLPRegressor(max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred
    }

    # Save predictions
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    pred_df.to_csv(f'Results/prediction_{name}_Model.csv', index=False)
    
    print(f"\n{name} Results:")
    print(f"MAE: {results[name]['mae']:.4f}")
    print(f"RMSE: {results[name]['rmse']:.4f}")
    print(f"R2 Score: {results[name]['r2']:.4f}")

# Model Visualizations

# Compare model performances
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [res['mae'] for res in results.values()],
    'RMSE': [res['rmse'] for res in results.values()],
    'R2': [res['r2'] for res in results.values()]
}).set_index('Model')

plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
sns.barplot(x=metrics_df.index, y='MAE', hue=metrics_df.index, data=metrics_df, palette='viridis', legend=False)
plt.title('Model MAE Comparison')

plt.subplot(2, 2, 2)
sns.barplot(x=metrics_df.index, y='RMSE', hue=metrics_df.index, data=metrics_df, palette='viridis', legend=False)
plt.title('Model RMSE Comparison')

plt.subplot(2, 2, 3)
sns.barplot(x=metrics_df.index, y='R2', hue=metrics_df.index, data=metrics_df, palette='viridis', legend=False)
plt.title('Model R2 Score Comparison')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Plot actual vs predicted for all models
plt.figure(figsize=(20, 15))
for i, (name, result) in enumerate(results.items()):
    plt.subplot(3, 3, i+1)
    plt.scatter(y_test, result['predictions'], alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name}\nR2: {result["r2"]:.2f}')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(metrics_df.T, 
            annot=True, 
            fmt=".2f", 
            cmap="Purples",
            linewidths=.5,
            cbar_kws={'label': 'Score'})

plt.title('Model Performance Comparison Heatmap', pad=20, fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('model_performance_heatmap.png', bbox_inches='tight', dpi=300)
plt.show()