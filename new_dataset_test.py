from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
import numpy as np
from urllib.parse import unquote
import pandas as pd
import pickle
import urllib.parse as parse
import pickle

np.random.seed(42)

# Filenames for models
filename1 = 'lib/DecisionTreeClassifier.sav'
filename2 = 'lib/SVC.sav'
filename3 = 'lib/GaussianNB.sav'
filename4 = 'lib/KNeighborsClassifier.sav'
filename5 = 'lib/RandomForestClassifier.sav'
filename6 = 'lib/MLPClassifier.sav'

# Load the models from disk
loaded_model1 = pickle.load(open(filename1, 'rb'))
loaded_model2 = pickle.load(open(filename2, 'rb'))
loaded_model3 = pickle.load(open(filename3, 'rb'))
loaded_model4 = pickle.load(open(filename4, 'rb'))
loaded_model5 = pickle.load(open(filename5, 'rb'))
loaded_model6 = pickle.load(open(filename6, 'rb'))
model = Doc2Vec.load("lib/d2v.model")

# Create a function to convert an array of query strings to a set of features
def getVec(text):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(text)]
    max_epochs = 25
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm=1)
    model.build_vocab(tagged_data)
    print("Building the sample vector model...")
    features = []
    for epoch in range(max_epochs):
        #print('Doc2Vec Iteration {0}'.format(epoch))
        print("*", sep=' ', end='', flush=True)
        model.random.seed(42)
        model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("lib/d2v.model")
    print()
    print("Model Saved")
    for i, line in enumerate(text):
        featureVec = [model.dv[i]]
        lineDecode = unquote(line)
        lineDecode = lineDecode.replace(" ", "")
        lowerStr = str(lineDecode).lower()
        #print("X"+str(i)+"=> "+line)
        # We could expand the features
        # https://websitesetup.org/javascript-cheat-sheet/
        # https://owasp.org/www-community/xss-filter-evasion-cheatsheet
        # https://html5sec.org/
        
        # add feature for malicious HTML tag count
        feature1 = int(lowerStr.count('<link'))
        feature1 += int(lowerStr.count('<object'))
        feature1 += int(lowerStr.count('<form'))
        feature1 += int(lowerStr.count('<embed'))
        feature1 += int(lowerStr.count('<ilayer'))
        feature1 += int(lowerStr.count('<layer'))
        feature1 += int(lowerStr.count('<style'))
        feature1 += int(lowerStr.count('<applet'))
        feature1 += int(lowerStr.count('<meta'))
        feature1 += int(lowerStr.count('<img'))
        feature1 += int(lowerStr.count('<iframe'))
        feature1 += int(lowerStr.count('<input'))
        feature1 += int(lowerStr.count('<body'))
        feature1 += int(lowerStr.count('<video'))
        feature1 += int(lowerStr.count('<button'))
        feature1 += int(lowerStr.count('<math'))
        feature1 += int(lowerStr.count('<picture'))
        feature1 += int(lowerStr.count('<map'))
        feature1 += int(lowerStr.count('<svg'))
        feature1 += int(lowerStr.count('<div'))
        feature1 += int(lowerStr.count('<a'))
        feature1 += int(lowerStr.count('<details'))
        feature1 += int(lowerStr.count('<frameset'))
        feature1 += int(lowerStr.count('<table'))
        feature1 += int(lowerStr.count('<comment'))
        feature1 += int(lowerStr.count('<base'))
        feature1 += int(lowerStr.count('<image'))
        # add feature for malicious method/event count
        feature2 = int(lowerStr.count('exec'))
        feature2 += int(lowerStr.count('fromcharcode'))
        feature2 += int(lowerStr.count('eval'))
        feature2 += int(lowerStr.count('alert'))
        feature2 += int(lowerStr.count('getelementsbytagname'))
        feature2 += int(lowerStr.count('write'))
        feature2 += int(lowerStr.count('unescape'))
        feature2 += int(lowerStr.count('escape'))
        feature2 += int(lowerStr.count('prompt'))
        feature2 += int(lowerStr.count('onload'))
        feature2 += int(lowerStr.count('onclick'))
        feature2 += int(lowerStr.count('onerror'))
        feature2 += int(lowerStr.count('onpage'))
        feature2 += int(lowerStr.count('confirm'))
        feature2 += int(lowerStr.count('marquee'))
        # add feature for ".js" count
        feature3 = int(lowerStr.count('.js'))
        # add feature for "javascript" count
        feature4 = int(lowerStr.count('javascript'))
        # add feature for length of the string
        feature5 = int(len(lowerStr))
        # add feature for "<script"  count
        feature6 = int(lowerStr.count('<script'))
        feature6 += int(lowerStr.count('&lt;script'))
        feature6 += int(lowerStr.count('%3cscript'))
        feature6 += int(lowerStr.count('%3c%73%63%72%69%70%74'))
        # add feature for special character count
        feature7 = int(lowerStr.count('&'))
        feature7 += int(lowerStr.count('<'))
        feature7 += int(lowerStr.count('>'))
        feature7 += int(lowerStr.count('"'))
        feature7 += int(lowerStr.count('\''))
        feature7 += int(lowerStr.count('/'))
        feature7 += int(lowerStr.count('%'))
        feature7 += int(lowerStr.count('*'))
        feature7 += int(lowerStr.count(';'))
        feature7 += int(lowerStr.count('+'))
        feature7 += int(lowerStr.count('='))
        feature7 += int(lowerStr.count('%3C'))
        # add feature for http count
        feature8 = int(lowerStr.count('http'))
        
        # append the features
        featureVec = np.append(featureVec,feature1)
        #featureVec = np.append(featureVec,feature2)
        featureVec = np.append(featureVec,feature3)
        featureVec = np.append(featureVec,feature4)
        featureVec = np.append(featureVec,feature5)
        featureVec = np.append(featureVec,feature6)
        featureVec = np.append(featureVec,feature7)
        #featureVec = np.append(featureVec,feature8)
        #print(featureVec)
        features.append(featureVec)
    return features
# Load and prepare the dataset
df = pd.read_csv('datasetClean.csv')
# Access the 'Sentence' column
sentences = df["Sentence"]
# Convert the Series to a list of strings
X = getVec(sentences.tolist())
y = df['Type'].apply(lambda x: 1 if x == 'Malicious' else 0).values  # Convert labels to 1 for Malicious and 0 for Benign

# Make predictions
ynew1 = loaded_model1.predict(X)
ynew2 = loaded_model2.predict(X)
ynew3 = loaded_model3.predict(X)
ynew4 = loaded_model4.predict(X)
ynew5 = loaded_model5.predict(X)
ynew6 = loaded_model6.predict(X)

# Aggregate predictions
def aggregate_predictions(*predictions):
    aggregated = []
    for i in range(len(predictions[0])):
        score = (0.175 * predictions[0][i] + 
                 0.15 * predictions[1][i] + 
                 0.05 * predictions[2][i] + 
                 0.075 * predictions[3][i] + 
                 0.25 * predictions[4][i] + 
                 0.3 * predictions[5][i])
        aggregated.append(1 if score >= 0.5 else 0)
    return aggregated

y_pred = aggregate_predictions(ynew1, ynew2, ynew3, ynew4, ynew5, ynew6)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

# Print results
xssCount = sum(y_pred)
notXssCount = len(y_pred) - xssCount

print("\n*------------- RESULTS -------------*")
print(f"Accuracy: {accuracy:.2f}")
print(f"\033[1;31;1mXSS\033[0;0m => {xssCount}")
print(f"\033[1;32;1mNOT XSS\033[0;0m => {notXssCount}")
