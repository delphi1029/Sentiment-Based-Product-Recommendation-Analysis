import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb


# Load the necessary models and data
user_final_rating = pickle.load(open("pickle/user_final_rating.pkl", "rb"))
df_final = pickle.load(open("pickle/cleaned-data.pkl", "rb"))
tfidf = pickle.load(open("pickle/tfidf-vectorizer.pkl", "rb"))
xgb = pickle.load(open("pickle/sentiment-classification-xg-boost.pkl", "rb"))

def get_top_products_for_user(user_name, user_final_rating):
    if user_name not in user_final_rating.index:
        return None
    return list(user_final_rating.loc[user_name].sort_values(ascending=False)[:20].index)

def filter_and_prepare_reviews(top_products, df_final):
    df = df_final[df_final.name.isin(top_products)].drop_duplicates(subset=['cleaned_review'])
    return df

def extract_features(df, tfidf):
    X = tfidf.transform(df['cleaned_review'])
    X_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())
    X_num = df[['review_length']].reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)
    return pd.concat([X_df, X_num], axis=1)

def predict_sentiment(df, features, xgb):
    df = df.copy()
    df['predicted_sentiment'] = xgb.predict(features)
    df['positive_sentiment'] = df['predicted_sentiment'].apply(lambda x: 1 if x == 1 else 0)
    return df

def aggregate_sentiment(df):
    pred_df = df.groupby(by='name').sum()
    pred_df = pred_df.rename(columns={'positive_sentiment': 'pos_sent_count'})
    pred_df['total_sent_count'] = df.groupby(by='name')['predicted_sentiment'].count()
    pred_df['pos_sent_percentage'] = np.round(pred_df['pos_sent_count'] / pred_df['total_sent_count'] * 100, 2)
    return pred_df.reset_index()

def product_recommendations_user(user_name):
    top_products = get_top_products_for_user(user_name, user_final_rating)
    if top_products is None:
        return f"The user '{user_name}' does not exist. Please provide a valid user name."
    df_reviews = filter_and_prepare_reviews(top_products, df_final)
    if df_reviews.empty:
        return "No recommendations found for this user."
    features = extract_features(df_reviews, tfidf)
    df_with_sentiment = predict_sentiment(df_reviews, features, xgb)
    pred_df = aggregate_sentiment(df_with_sentiment)
    return pred_df.sort_values(by="pos_sent_percentage", ascending=False)[:5][["name", "pos_sent_percentage"]]
