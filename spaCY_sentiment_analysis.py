import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")


file_path = 'sentiment_examples.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    sentiment_texts = file.readlines()

# Initialize empty lists to store the results
token_lists = []
filtered_token_lists = []
pos_tag_lists = []
ner_lists = []
# Process each sentiment example using spaCy and store the results
for sentiment_text in sentiment_texts:
    #print(sentiment_text)

    doc = nlp(sentiment_text.strip())  # Strip any leading/trailing whitespace
    #print(sentiment_texts)
   



    # Tokenization
    tokens = []
    for token in doc:
        tokens.append(token.text)
    token_lists.append(tokens)

    # tokens = [token.text for token in doc]  # Extract tokens from the processed text
    # token_lists.append(tokens)  # Append tokens list to token_lists


    # Stop Word Removal filter
    filtered_tokens = []
    for token in doc:
        if not token.is_stop:
            filtered_tokens.append(token.text)
    filtered_token_lists.append(filtered_tokens)

    #print(filtered_token_lists)
    # Part-of-Speech Tagging (POS tagging)
    pos_tags=[]
    for token in doc:
        pos_tags.append((token.text,token.pos_))
    pos_tag_lists.append(pos_tags)


    # Named Entity Recognition (NER)
    ner_entities=[]

    for ent in doc.ents:
        print(ent.text)
        ner_entities.append((ent.text, ent.label_))


    ner_lists.append(ner_entities)
        #print(ner_lists)
  

# Create a DataFrame to organize the results
results_df = pd.DataFrame({
    'Sentiment Example': sentiment_texts,
    'Tokens': token_lists,
    'Filtered Tokens': filtered_token_lists,
    'POS Tags': pos_tag_lists,
    'Named Entities': ner_lists
})

# Display the DataFrame
print(results_df)
print(results_df['POS Tags'])
print(results_df['Named Entities'])


