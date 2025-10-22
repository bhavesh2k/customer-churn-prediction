import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"data/Telco-Customer-Churn.csv")
df.dropna(inplace=True)

# Encode target
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

# Convert categorical variables to dummy variables
# After converting df to dummies
df = pd.get_dummies(df, drop_first=True)
feature_names = df.drop("Churn", axis=1).columns.tolist()
pickle.dump(feature_names, open(r"model/feature_names.pkl", "wb"))


# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
pickle.dump(model, open(r"model/churn_model.pkl", "wb"))
pickle.dump(scaler, open(r"model/scaler.pkl", "wb"))
