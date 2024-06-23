#import libraries
import pandas as pd
#added in due to iteritems being depreciated
pd.DataFrame.iteritems = pd.DataFrame.items
import plotly.express as px
import spacy
import csv
from textblob import TextBlob

#read in customer feedback file
nlp = spacy.load("en_core_web_sm")
file_path = 'feedback_data.csv'
with open(file_path, 'r', encoding='utf-8') as file:
    feedback_data = file.readlines()

#create output file to store the results
output_csv_path = "feedback_analysis_results.csv"
# Prepare CSV header
csv_header = ["Feedback Index", "Sentiment Polarity", "Sentiment Subjectivity", "Named Entities", "Preferred Contact Method"]

# Open CSV file for writing
with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
    # Create CSV writer
    csv_writer=csv.writer(csv_file)

    # Write the header
    csv_writer.writerow(csv_header)


#analyse each feedback


#process using spacy
index =0
for sentiment_text in feedback_data:
    index+=1
    #print(sentiment_text)

     
    print(sentiment_text.strip())
    doc = nlp(sentiment_text)

#use textblob library to perfom sentiment analysis
    blob = TextBlob(sentiment_text)
    reviews_polarity= round(blob.sentiment.polarity,2)
    reviews_subjectivity= round(blob.sentiment.subjectivity,2)
    print(f"polairity score: {reviews_polarity}, subjectivity score: {reviews_subjectivity}")
   
#extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print("Named Entities:", entities)

#prefered contact method
    word=".com"in sentiment_text.lower()
    if word==True:
        print("preferred contact type is email")
        type=("email")
    else:
        print("preferred contact type is chat")
        type=("chat")

#load results to dataframe
    with open(output_csv_path, "a", newline="", encoding="utf-8") as csv_file:
        csv_writer=csv.writer(csv_file)
        csv_writer.writerow([index, reviews_polarity, reviews_subjectivity, entities, type])
        
df= pd.read_csv(output_csv_path)
#display the head
print(df.head)

contact_types =df["Preferred Contact Method"].value_counts()
#print(contact_types.values)

#make a plot of prefered contact method
fig = px.bar(x=contact_types.index, y=contact_types.values, color=contact_types.index, labels={'x':'Contact Method', 'y':'Count'}, title='Preferred Contact Method Counts')
fig.show()

#chart the sentiment scores (both) on a pair plot
fig2 = px.scatter_matrix(df, dimensions=['Sentiment Polarity', 'Sentiment Subjectivity'],
                        title='Pair Plot: Pairwise relationships')
fig2.show()