
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from transformers import pipeline
#import csv

# Load CSV file into a DataFrame without header
df = pd.read_csv("ch4_feedback_data.csv", header=None)
print(df.head())

from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
#text = df
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)



sentiment_analysis_bert = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_results_bert = []
for index, row in df.iterrows():
    text = row[0]  # Access the first (and only) column in each row
    bert_result = sentiment_analysis_bert(text)[0]
    sentiment_label_bert = bert_result['label']
    sentiment_score_bert = bert_result['score']
    sentiment_results_bert.append({"Text": text, "Sentiment Score": sentiment_score_bert, "Sentiment Label": sentiment_label_bert})


# Iterate over the sentiment results and print each entry
for result in sentiment_results_bert:
    print(f"Text: {result['Text']}")
    print(f"Sentiment Score: {result['Sentiment Score']}")
    print(f"Sentiment Label: {result['Sentiment Label']}")
    print()  # Add a blank line for readability

# Convert DistilBERT sentiment results to DataFrame
sentiment_df_bert = pd.DataFrame(sentiment_results_bert)

# Save DistilBERT sentiment results to a new CSV file
sentiment_df_bert.to_csv("sentiment_results_bert.csv", index=False)

print(sentiment_df_bert.head())

#chart
rating_counts = sentiment_df_bert['Sentiment Label'].value_counts()
fig = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values,
             labels={'x':'Sentiment label', 'y':'Frequency Count'},
                     title='Bar Chart of Sentiment Labels')
fig.show()
