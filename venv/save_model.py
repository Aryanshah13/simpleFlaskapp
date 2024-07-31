import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv(r"C:\Users\shaha\Downloads\processed-data.csv")

# Drop unnecessary columns and rename the target column
df = df.drop(df.columns[[4, 5]], axis=1)
df.rename(columns={'Severity_None': 'Target'}, inplace=True)

# Prepare data for training and testing
x = df.drop(columns=['Target'])
y = df['Target']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)

# Train the model
model.fit(x_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

