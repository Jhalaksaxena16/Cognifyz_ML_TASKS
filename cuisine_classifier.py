import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("Dataset .csv")
df = df[['Cuisines', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes', 'City']]
df.dropna(inplace=True)

le_cuisine = LabelEncoder()
le_city = LabelEncoder()
df['Cuisines'] = le_cuisine.fit_transform(df['Cuisines'])
df['City'] = le_city.fit_transform(df['City'])

X = df[['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes', 'City']]
y = df['Cuisines']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
