import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Dataset .csv")
df = df[['Restaurant Name', 'Cuisines', 'Price range']].dropna()
df['Features'] = df['Cuisines'] + ' ' + df['Price range'].astype(str)

vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(df['Features'])

def recommend_restaurants(cuisine, price_range):
    user_input = f"{cuisine} {price_range}"
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, feature_matrix).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    return df.iloc[top_indices][['Restaurant Name', 'Cuisines', 'Price range']]

print("Top 5 Restaurant Recommendations for: North Indian, Price 2")
results = recommend_restaurants("North Indian", 2)
print(results.to_string(index=False))
