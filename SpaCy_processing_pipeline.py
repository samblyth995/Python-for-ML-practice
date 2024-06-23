import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

text="a customer in new york city wants to give a review"
doc=nlp(text)
#Processing Pipeline Order

#Tokenization
print("Tokenization:")
for token in doc:
    print(token.text)
#Stop Words Removal
print("Filtered Tokens (without stop words):")
filtered_tokens = [token.text for token in doc if not token.is_stop]
print(filtered_tokens)

#POS Tagging
    # spaCy can parse and tag a given Doc. This is where the trained pipeline and its statistical models come in, which enable spaCy to make predictions of which tag or label most likely applies in this context. A trained component includes binary data that is produced by showing a system enough examples for it to make predictions that generalize across the language – for example, a word following “the” in English is most likely a noun.
    # Linguistic annotations are available as Token attributes. Like many NLP libraries, spaCy encodes all strings to hash values to reduce memory usage and improve efficiency. So to get the readable string representation of an attribute, we need to add an underscore _ to its name:
print("Part-of-Speech Tagging (POS):")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
    print("\n")

#Dependence Parsing
#Lemmatization
print("Lemmatization:")
lemmatized_tokens = [token.lemma_ for token in doc if not token.is_punct]
print(lemmatized_tokens)
#Named Entity Recognition
print("Named Entity Recognition (NER):")
for ent in doc.ents:
    print(ent.text, ent.label)
print("\n") 

