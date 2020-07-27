#!/usr/bin/env python
# coding: utf-8

# # Importations des bases

# In[1]:


import pandas as pd


# In[ ]:


#Lien vers la page kaggle où les données sont disponibles
# https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset


# In[2]:


fake = pd.read_csv("D:/Cours_AMSE/Machine_Learning/Fake.csv", header = 0 ,sep=",")


# In[3]:


true = pd.read_csv("D:/Cours_AMSE/Machine_Learning/True.csv", header = 0 ,sep=",")


# In[4]:


print(len(fake))
print(len(true))


# In[5]:


fake.iloc[0,1]


# In[6]:


fake.head()


# In[7]:


true.head()


# # Pré traitement des données

# In[8]:


import re


# In[9]:


#suppresion des caractères indésirables (@..., #...hashtags, les url, les auteurs à la fin: Photo by... )

for i in range (0,len(fake)):
    fake.iloc[i,1] = re.sub('.@\w*. ','', fake.iloc[i,1])

for i in range (0,len(true)):
    true.iloc[i,1] = re.sub('.@\w*. ','', true.iloc[i,1])

for i in range (0,len(fake)):
    fake.iloc[i,1] = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', fake.iloc[i,1], flags=re.MULTILINE)

for i in range (0,len(true)):
    true.iloc[i,1] = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', true.iloc[i,1], flags=re.MULTILINE)

for i in range (0,len(fake)):
    fake.iloc[i,1] = re.sub("pic.twitter.com.\w*", "",fake.iloc[i,1])

for i in range (0,len(true)):
    true.iloc[i,1] = re.sub("pic.twitter.com.\w*", "",true.iloc[i,1])
    
for i in range (0,len(fake)):
    fake.iloc[i,1] = re.sub('Featured image.*|Photo by.*|#\w*',"",fake.iloc[i,1])

for i in range (0,len(true)):
    true.iloc[i,1] = re.sub('Featured image.*|Photo by.*|#\w*',"",true.iloc[i,1])


# # Description des données

# In[10]:


'''En vue de donner les fonctions des mots (tagging) et d'identifier les mots les plus fréquents, on fusionne
les textes dans chaque base de données''' 

S_true= ""
for i in range (0,len(true)):
    S_true = S_true + true.iloc[i,1]


# In[11]:


S_fake= ""
for i in range (0,len(fake)):
    S_fake = S_fake + fake.iloc[i,1]


# In[12]:


#importation du package natural langage toolkit et de package pour tokenizer (séparer le texte en mots)

import nltk
nltk.download("punkt")


# In[13]:


##importation du package pour identifier les fonction des mots

nltk.download("averaged_perceptron_tagger")
from nltk.tag import pos_tag


# In[14]:


tokenizer = nltk.RegexpTokenizer(r"\w+") ## on choisit une méthode de tokenization qui supprime la ponctuation
tokens_fake = tokenizer.tokenize(S_fake.lower()) ## tokenizer en suprimant la ponctuation et meetre en minuscule le texte
S_fake = nltk.Text(tokens_fake) 
tags_f = nltk.pos_tag(S_fake)  #attribuer la fonction de chaque mot


# In[15]:


##importer un package qui compte les mots 

from collections import Counter
counts_tag_fake = Counter(tag for word,tag in tags_f)


# In[16]:


##effectuer le compte dans la base fake

total_f = sum(counts_tag_fake.values())
d_f= dict((word, float(count)/total_f) for word,count in counts_tag_fake.items())


# In[17]:


tokens_true = tokenizer.tokenize(S_true.lower())
S_true = nltk.Text(tokens_true)
tags_t = nltk.pos_tag(S_true)


# In[18]:


counts_tag_true = Counter(tag for word,tag in tags_t)


# In[19]:


##faire le total de chaque fonction de mots : i.e : compter tous les noms 

total_t = sum(counts_tag_true.values())
d_t= dict((word, float(count)/total_f) for word,count in counts_tag_true.items())


# In[20]:


##ici on regroupe toutes les formes de mots de fonctions(function word: pronoms,prépositions...) d'adjectif, d'adverbes, de noms, de verbe



fw_t = d_t["IN"]+ d_t["PRP"]+d_t["PRP$"]+d_t["WP"]+d_t["WP$"]+d_t["DT"]+d_t["PDT"]+d_t["MD"]+d_t["WDT"]+d_t["UH"]+d_t["POS"]+d_t["TO"]+d_t["EX"]
adj_t = d_t["JJ"]+ d_t["JJR"]+d_t["JJS"]
adv_t= d_t["RB"]+ d_t["RBR"]+d_t["RBS"]+d_t["WRB"]
nn_t = d_t["NN"]+d_t["NNS"]+d_t["NNP"]+d_t["NNPS"]
v_t = d_t["RP"]+d_t["VB"]+d_t["VBD"]+d_t["VBG"]+d_t["VBN"]+d_t["VBP"]+d_t["VBZ"]

fw_f = d_f["IN"]+d_f["PRP"]+d_f["PRP$"]+d_f["WP"]+d_f["WP$"]+d_f["DT"]+d_f["PDT"]+d_f["MD"]+d_f["WDT"]+d_f["UH"]+d_f["POS"]+d_f["TO"]+d_f["EX"]
adj_f = d_f["JJ"]+d_f["JJR"]+d_f["JJS"]
adv_f= d_f["RB"]+d_f["RBR"]+d_f["RBS"]+d_f["WRB"]
nn_f = d_f["NN"]+d_f["NNS"]+d_f["NNP"]+d_f["NNPS"]
v_f = d_f["RP"]+d_f["VB"]+d_f["VBD"]+d_f["VBG"]+d_f["VBN"]+d_f["VBP"]+d_f["VBZ"]

'''function word: préposition: in
pronoun: PRR
possessive pronon : PRP$
possessive wh-pronoun whose :WP$ 
wh-pronoun who, what: WP
cojonction de corrd : CC
determiner : DT 
predeterminer : PDT
modal:MD
wh-determiner : WDT
interjection : UH
possessive ending POS
TO : to
existential there: EX 

Content word : Content words are words with specific meanings, such as nouns, adjectives, adverbs, and main verbs

adjectives:JJ
adjective, comparative: JJR 
adjective, superlative: JJS 
wh-adverb where, when : WRB 

noun, singular: NN 
noun plural : NNS
proper noun, singular : NNP
proper noun, plural ‘Americans’ : NNPS

adverb very, silently, :RB 
adverb, comparative better: RBR 
adverb, superlative best: RBS 

particle give up :RP 
verb, base form : VB
verb, past tense, took :VBD
verb, gerund/present participle taking:VBG 

verb, past participle is taken:VBN 
verb, sing. present, known-3d take:VBP 
verb, 3rd person sing. present takes:VBZ'''
 


# In[21]:


import matplotlib.pyplot as plt
import numpy as np


# In[22]:


##on représente toutes ces types de mots sous formes de barplot

group = ["Fonction word", "Adjectives", "Adverbs", "Nouns","Verbs"]
bar_t = [fw_t,adj_t,adv_t,nn_t,v_t]
bar_f = [fw_f,adj_f,adv_f,nn_f,v_f]
# Position sur l'axe des x pour chaque étiquette
position = np.arange(len(group))
# Largeur des barres
largeur = .35
# Création de la figure et d'un set de sous-graphiques
fig, ax = plt.subplots()
r1 = ax.bar(position - largeur/2,bar_t , largeur, label = "True")
r2 = ax.bar(position + largeur/2,bar_f , largeur,label = "Fake")
# Modification des marques sur l'axe des x et de leurs étiquettes
ax.set_xticks(position)
ax.set_xticklabels(group)
plt.legend(loc = "upper center")
plt.savefig('description.png')


# In[23]:


##importer le package pour supprimer les stopwords et tokenizer

nltk.download("stopwords")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[24]:


##spécifier qu'on veut des stopword anglais
stop_words = set(stopwords.words('english'))


# In[25]:


##filtrer les stopwords

stop_words = set(stopwords.words('english'))
filtered_fake = [] 
  
for w in S_fake: 
    if w not in stop_words: 
        filtered_fake.append(w)


# In[26]:


import collections
counts_nsw_f = collections.Counter(filtered_fake)


# In[27]:


filtered_true = [] 
  
for w in S_true: 
    if w not in stop_words: 
        filtered_true.append(w)


# In[28]:


##compter combien de fois chaque mot revient

counts_nsw_t = collections.Counter(filtered_true)


# In[29]:


##regrouper dans dans un dataframe les 10 mots les  plus fréquents avec leurs effectifs respectifs

count_mc_f = pd.DataFrame(counts_nsw_f.most_common(10),
                             columns=['words', 'count'])
count_mc_f.head()


# In[30]:


count_mc_t = pd.DataFrame(counts_nsw_t.most_common(10),
                             columns=['words', 'count'])
count_mc_t.head()


# # Visualisation des données

# In[31]:


##représentation graphique des effectifs des mots les  plus fréquents dans la base des fake news

fig, ax = plt.subplots(figsize=(12, 12))

# Plot horizontal bar graph
count_mc_f.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Les 10 mots les plus fréquents (sans les stopwords)")

plt.show()

plt.savefig('fake_words.png')


# In[32]:


##représentation graphique des effectifs des mots les  plus fréquents dans la base des true news

fig, ax = plt.subplots(figsize=(12, 12))

# Plot horizontal bar graph
count_mc_t.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="darkturquoise")

ax.set_title("Les 10 mots les plus fréquents (sans les stopwords)")

plt.show()
plt.savefig('truee_words.png')


# # Préparation des données pour la classification 

# In[33]:


#ajoutons à chaque base, la variable  Nature pour indiquer si l'info est vraie ou fausse

fake['Nature'] = "faux"
true['Nature'] = "vrai"


# In[34]:


#fusionner les deux bases 

mabase = fake.append(true,ignore_index=True)


# In[35]:


##tokenization and lowercasing

mabase['tokenized_text'] = mabase.apply(lambda row: tokenizer.tokenize((row['text']).lower()) , axis=1)


# In[36]:


##stopwords removing

mabase['wst_text'] = mabase['tokenized_text'].apply(lambda x: [item for item in x if item not in stop_words])


# In[37]:


#importer la base pour la lemmatisation (stemming)

from nltk.stem import PorterStemmer
ps = PorterStemmer()


# In[38]:


##stemming

mabase['stem_text'] = mabase['wst_text'].apply(lambda x: [ps.stem(y) for y in x])


# In[39]:


list(mabase)


# In[40]:


mabase.iloc[0,7]


# In[41]:


## lorsqu'on passe ensuite à la classification en utilisant directement cette colonne 'stem_text' une erreur est générée car 
## car 'stem_text' contient des listes. pour éviter cela nous ramenons les listes contenues dans cette colonne sous forme de 
## de texte. Pour cela nous créons une nouvelle variable qui va contenir le texte lemmatisé 

mabase['re_stem_text']=""


# In[42]:


for i in range (0,len(mabase)):
    mabase.iloc[i,8] = " ".join(mabase.iloc[i,7])


# In[43]:


x = mabase['re_stem_text']
y = mabase.Nature
len(y)


# # Les différentes méthodes de classification

# In[44]:


##construction d'une fonction de graphique pour les matrices de confusion
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",verticalalignment = 'bottom',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[45]:


##importation des differents packages nécessaires

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


# In[46]:


##méthode de Multinomial Naive Bayesian. Pipeline va permettre de réalisation la tranformartion des données avec tfidf et
##de d'apprêter la methode de classification

clf_mlb = Pipeline([('vect', TfidfVectorizer(stop_words='english',lowercase=True,ngram_range=(1,1))), ('mlb',MultinomialNB())])


# In[47]:


from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression


# In[63]:


#### méthode de decision tree

clf_dt = Pipeline([('vect', TfidfVectorizer(stop_words='english',lowercase=True,ngram_range=(1,1))), ('dt',tree.DecisionTreeClassifier())]) 
score_dt = cross_val_score(clf_dt,x,y,cv=10,scoring='accuracy')
print(score_dt)
print("Accuracy: %0.2f (+/- %0.2f)" % (score_dt.mean(), score_dt.std() * 2))
y_pred = cross_val_predict(clf_dt,x , y, cv=10)
conf_mat = confusion_matrix(y, y_pred,labels=['faux', 'vrai'])
plot_confusion_matrix(conf_mat, classes=['faux', 'vrai']) ##dans la fonction de création du graphique pour la matrice de confuision
##mettre cmap=plt.cm.Purples et color="white"
plt.savefig('dt_mat.png')


# In[66]:


#### regression logistic

clf_lr = Pipeline([('vect', TfidfVectorizer(stop_words='english',lowercase=True,ngram_range=(1,1))), ('lr',LogisticRegression())]) 
score_lr = cross_val_score(clf_lr,x,y,cv=10,scoring='accuracy')
print(score_lr)
print("Accuracy: %0.2f (+/- %0.2f)" % (score_lr.mean(), score_lr.std() * 2))
y_pred = cross_val_predict(clf_lr,x , y, cv=10)
conf_mat = confusion_matrix(y, y_pred,labels=['faux', 'vrai'])
plot_confusion_matrix(conf_mat, classes=['faux', 'vrai'])  ##dans la fonction de création du graphique pour la matrice de confuision
##mettre cmap=plt.cm.Greys et color="white"
plt.savefig('lr_mat.png')


# In[68]:


#### méthode de PassiveAggrssiveClassifier


clf_pac = Pipeline([('vect', TfidfVectorizer(stop_words='english',lowercase=True,ngram_range=(1,1))), ('pac',PassiveAggressiveClassifier())]) 
score_pac = cross_val_score(clf_pac,x,y,cv=10,scoring='accuracy')
print(score_pac)
y_pred = cross_val_predict(clf_pac,x, y, cv=10)
conf_mat = confusion_matrix(y, y_pred,labels=['faux', 'vrai'])
plot_confusion_matrix(conf_mat, classes=['faux', 'vrai'])##dans la fonction de création du graphique pour la matrice de confuision
##mettre cmap=plt.cm.Blues et color="white"
plt.savefig('pac_mat.png')


# In[69]:


print("Accuracy: %0.2f (+/- %0.2f)" % (score_pac.mean(), score_pac.std() * 2))


# # Pour aller plus loin, méthode LDA pour la recheche des sujets les plus fréquents

# In[ ]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2,
                                   max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(mabase['re_stem_text'])


# In[89]:


from sklearn.decomposition import NMF, LatentDirichletAllocation


# In[90]:


## ici on essaie de mettre en oeuvre la méthode LDA pour la détection de thématiques, afin d'aller au delà de l'objectif de 
## classification et pour mieux comprendre la base de données dont nous disposons. Il faudra réexcuter cette section pour voir
## le graohique interactif 

lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)


# In[91]:


tf_feature_names = tfidf_vectorizer.get_feature_names()
tf_feature_names[100:103]


# In[92]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[- n_top_words - 1:][::-1]]))
    print()


# In[93]:


##thématiques les plus fréquentes

lda.fit(tfidf)
print_top_words(lda, tf_feature_names, 10)


# In[94]:


tr = lda.transform(tfidf)
tr[:5]


# In[95]:


pip install pyLDAvis


# In[96]:


import pyLDAvis.sklearn
pyLDAvis.enable_notebook()


# In[97]:


##représentation du graphique interactif

pyLDAvis.sklearn.prepare(lda, tfidf, tfidf_vectorizer)

