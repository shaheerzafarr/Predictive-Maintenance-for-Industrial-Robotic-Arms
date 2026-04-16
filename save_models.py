"""
Save trained models from the notebook for the UI app.
Run this ONCE after running all cells in mlr.ipynb.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('data/feature_time_48k_2048_load_1.csv')

# Create columns
def get_fault_type(f):
    if 'Normal' in f: return 'Normal'
    elif 'Ball' in f: return 'Ball'
    elif 'IR' in f: return 'InnerRace'
    elif 'OR' in f: return 'OuterRace'
    return 'Unknown'

def get_severity(f):
    if 'Normal' in f: return 0.0
    elif '007' in f: return 0.007
    elif '014' in f: return 0.014
    elif '021' in f: return 0.021
    return 0.0

df['fault_type'] = df['fault'].apply(get_fault_type)
df['severity'] = df['fault'].apply(get_severity)

feature_cols = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
X = df[feature_cols].values

le = LabelEncoder()
y = le.fit_transform(df['fault_type'])

# Scale
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Train models on FULL dataset for deployment
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_sc, y)

print("Training SVM...")
svm = SVC(C=10, gamma='scale', kernel='rbf', random_state=42, probability=True)
svm.fit(X_sc, y)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_sc, y)

print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
knn.fit(X_sc, y)

print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42)
dt.fit(X_sc, y)

# Severity regression (faulty samples only)
print("Training Regression...")
mask = df['severity'] > 0
X_reg = df.loc[mask, feature_cols].values
y_reg = df.loc[mask, 'severity'].values
scaler_reg = StandardScaler()
X_reg_sc = scaler_reg.fit_transform(X_reg)
reg_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg', LinearRegression())
])
reg_pipe.fit(X_reg_sc, y_reg)

# Q-table
Q = np.array([
    [ 10., -5., -20.],
    [ -5., 15., -10.],
    [-20., 10.,   5.],
    [-50.,  0.,  20.]
])
# Retrain Q-table quickly
n_states, n_actions = 4, 3
reward_matrix = np.array([
    [10, -5, -20], [-5, 15, -10], [-20, 10, 5], [-50, 0, 20]
])
Q = np.zeros((n_states, n_actions))
for _ in range(10000):
    state = np.random.randint(0, n_states)
    for _ in range(50):
        eps = 0.01
        action = np.random.randint(0, n_actions) if np.random.random() < eps else np.argmax(Q[state])
        reward = reward_matrix[state, action]
        if action == 2: ns = 0
        elif action == 1: ns = 0 if np.random.random() < 0.8 else max(0, state-1)
        else:
            if state == 0: ns = 1 if np.random.random() < 0.1 else 0
            else:
                r = np.random.random()
                ns = min(3, state+1) if r < 0.4 else (state if r < 0.8 else max(0, state-1))
        Q[state, action] += 0.1 * (reward + 0.95 * np.max(Q[ns]) - Q[state, action])
        state = ns

# Feature stats for sample generation
feature_stats = {}
for col in feature_cols:
    feature_stats[col] = {
        'mean': float(df[col].mean()),
        'std': float(df[col].std()),
        'min': float(df[col].min()),
        'max': float(df[col].max())
    }

# Save everything
models = {
    'scaler': scaler,
    'scaler_reg': scaler_reg,
    'label_encoder': le,
    'class_names': list(le.classes_),
    'feature_cols': feature_cols,
    'feature_stats': feature_stats,
    'models': {
        'Gradient Boosting': gb,
        'SVM': svm,
        'Random Forest': rf,
        'KNN': knn,
        'Decision Tree': dt
    },
    'regression': reg_pipe,
    'q_table': Q
}

with open('models/trained_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("\nAll models saved to models/trained_models.pkl!")
print(f"File size: {__import__('os').path.getsize('models/trained_models.pkl') / 1024 / 1024:.1f} MB")
