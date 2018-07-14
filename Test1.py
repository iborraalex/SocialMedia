# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:26:59 2018

@author: ALEJANDRO IBORRA
"""

from xml.dom import minidom
from os import listdir
from os.path import isfile, join
import time
import pandas as pd
import re
import nltk
import random
import emoji
from nltk.corpus import stopwords
import math
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from prettytable import PrettyTable
import json


start_time = time.time()

random.seed(1234)



prepos = {"a": 1, "ante": 1, "bajo": 1, "cabe": 1, "con": 1, "contra": 1, "de": 1, "desde": 1, "durante": 1,\
          "en": 1, "entre": 1, "hacia": 1, "hasta": 1, "mediante": 1, "para": 1, "por": 1, "segun": 1, "sin": 1, \
          "so": 1, "sobre": 1, "tras": 1, "versus": 1, "via": 1, "según": 1, "vía": 1 }



#nltk.download()



######## FUNCIONES VARIAS #############

def tweet_to_words( raw_tweet ):
    # The input is a single string (a tweet), and 
    # the output is a single string (a preprocessed tweet)
    #
    # Remove non-letters        
    letters_only = re.sub("[,.:;¿?!¡]", "", raw_tweet) 
    
    #
    # Convert to lower case, split into individual words
    #words = letters_only.lower().split()    
    words = letters_only.lower().split(" ") 
                             
    #
    # In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("spanish"))                
    # 
    # Remove stop words
    meaningful_words = [w for w in words if (not w in stops) and (not w.startswith("#")) and \
                        (not w.startswith("@"))  and (not w.startswith("http")) ]   
    #
    # Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   
    

def count_adjetivos ( tweet, c_masc, c_fem ):
    tweet_words = tweet.split()
    
    
    for w in tweet_words:
        if(w in dict_adjetivos):
            if ( w.endswith('a') or w.endswith('as') ):
                c_fem += 1
            elif (w.endswith('o') or w.endswith('os') or w.endswith('or') ):
                c_masc += 1
    
        
    
    return (c_masc, c_fem)


def count_emojis(str):
    return len(''.join(c for c in str if c in emoji.UNICODE_EMOJI))

def remove_emojis(str):
    return ''.join(c for c in str if c not in emoji.UNICODE_EMOJI)


def carga_listado_adjetivos():
    
    adjetivos = pd.read_csv("adjetivos-utf8.txt", header=None, names=["adj"])

    dict_adjetivos = dict(zip(adjetivos.adj,adjetivos.adj))   

    #Quitamos los dos raros del principio.
    del dict_adjetivos["VERDADERO"]
    del dict_adjetivos["FALSO"]

    print("Adjetivos cargados: ", len(adjetivos))
    #print("Adjetivo 0: ", adjetivos.adj[0])
    #print("Adjetivo 10: ", adjetivos.adj[10])
    print("--------------------------------------------")
    
    return dict_adjetivos


def cargar_truth_training():
    truth_training = pd.read_csv("training/truth.txt", header=None, names=["id", "gender", "country"], sep=";")

    print("Truth training cargado: ", len(truth_training))
    print("--------------------------------------------")

    return dict(zip(truth_training.id,truth_training.gender)) 


def contar_media_palabras ( id_texto ):
    count = 0
    num_tweets = 0
    for tweet in lista:
        if (tweet[0] == id_texto):
            count += len(tweet[1].split(" "))
            num_tweets += 1

    result = 0
    if(num_tweets > 0):
        result = count / num_tweets
    
    return (result, count)


def count_preposiciones ( id_texto, num_prep ):
    
    for tweet in lista:
        if (tweet[0] == id_texto):
            texto = tweet[1].split(" ")
            for text in texto:
                if text in prepos:
                    num_prep += 1

    return num_prep


def devuelve_vector(M, nombre_col=None, pre=2):
    vector_matriz=[]
    filas, columnas = M.shape
    #vector_matriz.append(nombre_col)
    for fila in range(filas):
        vf = M.getrow(fila)
        _, cind = vf.nonzero()
        vector_matriz.append([round(vf[0, c],pre) if c in cind else 0 for c in range(columnas)])
    return vector_matriz

def create_bag_words ():
    
    clean_train_tweets = []
    
    id_previo = ""
    bolsa_masc = []
    bolsa_fem = []


    for i in range( 0, len(lista)  ):
        # Call our function for each one, and add the result to the list of
        # clean reviews
    
        if(id_previo == ""):
            id_previo = lista[i][0]
    
        #Si cambiamos de documento...reseteo de contadores
        if( id_previo != lista[i][0] ):
                
            #Busqueda del GENDER dentro de TRUTH
            sexo = 0
            if ( truth_training.get(id_previo) == 'male' ):
                sexo = 1
            
            #Acumulamos las palabas de cada uno de los generos.
            for tweet in clean_train_tweets:
                if (sexo == 0):
                    bolsa_fem.append(remove_emojis(tweet))
                else:
                    bolsa_masc.append(remove_emojis(tweet))
        
        
            id_previo = lista[i][0]
            clean_train_tweets = []
        
    
    
        if( (i+1)%1000 == 0 ):
            print ("Bolsa palabras. Tweet ", i+1, " of ", len(lista) ) 
    
        clean_train_tweets.append( tweet_to_words( lista[i][1] ) )
    
    
    #Guardamos lo ultimo...
    sexo = 0
    if ( truth_training.get(id_previo) == 'male' ):
        sexo = 1
            
    for tweet in clean_train_tweets:
        if (sexo == 0):
            bolsa_fem.append(remove_emojis(tweet))
        else:
            bolsa_masc.append(remove_emojis(tweet))
    
        
    return bolsa_fem, bolsa_masc
    
    

def return_string_bag_words (listado):
    
    lista = []
    
    for i in range(0, len(listado)): 
        lista.append(listado[i])
        
    return ' '.join(lista)
    
    

#Para tf-idf
def tf(word, blob):
    #return (float)(blob.words.count(word)) / (float)(len(blob.words))
    return (float)(blob.words.count(word)) 

def n_containing(word, bloblist):
    return (float)(sum(1 for blob in bloblist if word in blob))

def idf(word, bloblist):
    return (float)(math.log(len(bloblist))) / (float)((1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return (float)((float)(tf(word, blob)) * (float)(idf(word, bloblist)))
    
    
    



####### CARGAMOS ADJETIVOS ###########
print("cargamos adjetivos")

dict_adjetivos = carga_listado_adjetivos()

truth_training = cargar_truth_training()


####### CARGA DE DOCUMENTOS XML #########

docsTraining = [f for f in listdir('training') if isfile(join('training', f))]

print("Num ficheros: ", len(docsTraining))
lista = []

# Iterando por todos los documentos
for i in range(0, len(docsTraining)):     # len(docsTraining)
    
    file = docsTraining[i]
    fileName = docsTraining[i][0:-4]

    #No procesamos el archivo de truth.
    if(fileName != "truth"):
        xmldoc = minidom.parse('training/' + file)
        itemlist = xmldoc.getElementsByTagName('document')


        #Cargamos los 100 tweets
        for j in range(0, len(itemlist)):
            tweet = itemlist[j].childNodes[0].nodeValue
            lista += [(fileName, tweet)]
            
  
print("Tweets cargados: ", len(lista))
print("--------------------------------------------")




######## CREACION BOLSA PALABRAS  ###############


bolsa_fem, bolsa_masc = create_bag_words()

num_features_bow = 1000                                            #ALTERAR

start_time = time.time()

#Female
vec_female = CountVectorizer(max_features=num_features_bow)
X_female = vec_female.fit_transform(bolsa_fem)
voca_female = vec_female.get_feature_names()

#Male
vec_male = CountVectorizer(max_features=num_features_bow)
X_male = vec_male.fit_transform(bolsa_masc)
voca_male = vec_male.get_feature_names()

print('Bag of Words creation [NEW]. It took {0:0.2f} seconds'.format(time.time() - start_time))



#######  PROCESAR  ##################

clean_train_tweets = []

lista_modelo = []
id_previo = ""
c_masc = 0
c_fem = 0
num_prep = 0
num_emojis = 0

for i in range( 0, len(lista)  ):    #Pruebas y verificaciones con 400 (4 docs)
    
    if(id_previo == ""):
        id_previo = lista[i][0]
    
    #Si cambiamos de documento...reseteo de contadores
    if( id_previo != lista[i][0] ):
        
        for tweet in clean_train_tweets:
            c_masc, c_fem = count_adjetivos ( tweet, c_masc, c_fem )
            num_emojis += count_emojis(tweet)
            
        
        #Busqueda del GENDER dentro de TRUTH
        sexo = 0
        if ( truth_training.get(id_previo) == 'male' ):
            sexo = 1
            
            
        #Contamos media de palabras por tweet para cada autor / fichero.
        count_media_palabras, count_total_palabras = contar_media_palabras (id_previo)
        
        #Contamos total de preposiciones usadas.
        num_prep = count_preposiciones ( id_previo, num_prep )
        
        
        
        tuple_modelo_final = (id_previo, count_media_palabras, num_emojis/count_media_palabras, num_prep/count_media_palabras, c_masc/count_media_palabras, c_fem/count_media_palabras, (int) (c_masc > c_fem))
        
        ##Procesamos la bolsa de PALABRAS
        todos_tweets_archivo_string = ' '.join(clean_train_tweets)
        todos_tweets_archivo = []
        todos_tweets_archivo.append(todos_tweets_archivo_string)
        
        # Aplicamos vocabulario mujer    
        Y=vec_female.transform(todos_tweets_archivo)
        vector_female = devuelve_vector(Y,voca_female)
        
        # Aplicamos vocabulario hombre
        Y=vec_male.transform(todos_tweets_archivo)
        vector_male = devuelve_vector(Y,voca_male)
        
        #Paso a lista para trabajar con los vocabularios 
        l_tuple_modelo_final = list(tuple_modelo_final)
        
        #Añado vocabulario mujer
        for x in vector_female[0]:
            l_tuple_modelo_final.append(x)
            
        #Añado vocabulario hombre
        for x in vector_male[0]:
            l_tuple_modelo_final.append(x)
         
        #Ultimo, la etiqueta del sexo
        l_tuple_modelo_final.append(sexo)
            
        #Convierto de nuevo a tupla
        tuple_modelo_final = tuple(l_tuple_modelo_final)
        
        
        #Guardamos para el modelo final
        lista_modelo += [tuple_modelo_final]
        
        
        id_previo = lista[i][0]
        c_masc = 0
        c_fem = 0
        num_prep = 0
        num_emojis = 0
        clean_train_tweets = []
        
    
    
    if( (i+1)%1000 == 0 ):
        print ("Tweet ", i+1, " of ", len(lista) ) 
    
    clean_train_tweets.append( tweet_to_words( lista[i][1] ) )
    
    
#Guardamos lo ULTIMO...
for tweet in clean_train_tweets:
    c_masc, c_fem = count_adjetivos ( tweet, c_masc, c_fem )
    num_emojis += count_emojis(tweet)
#Busqueda del GENDER dentro de TRUTH
sexo = 0
if ( truth_training.get(id_previo) == 'male' ):
    sexo = 1        
#Guardamos otro atributo para si masculino o femenino
count_media_palabras, count_total_palabras  = contar_media_palabras (id_previo)
num_prep = count_preposiciones ( id_previo, num_prep )
tuple_modelo_final = (id_previo, count_media_palabras, num_emojis/count_media_palabras, num_prep/count_media_palabras, c_masc/count_media_palabras, c_fem/count_media_palabras, (int) (c_masc > c_fem))

#Procesamos la bolsa de PALABRAS
todos_tweets_archivo_string = ' '.join(clean_train_tweets)
todos_tweets_archivo = []
todos_tweets_archivo.append(todos_tweets_archivo_string)

# Aplicamos vocabulario mujer    
Y=vec_female.transform(todos_tweets_archivo)
vector_female = devuelve_vector(Y,voca_female)

# Aplicamos vocabulario hombre
Y=vec_male.transform(todos_tweets_archivo)
vector_male = devuelve_vector(Y,voca_male)
#Paso a lista para trabajar con los vocabularios 
l_tuple_modelo_final = list(tuple_modelo_final)

#Añado vocabulario mujer
for x in vector_female[0]:
    l_tuple_modelo_final.append(x)

#Añado vocabulario hombre
for x in vector_male[0]:
    l_tuple_modelo_final.append(x)

#Ultimo, la etiqueta del sexo
l_tuple_modelo_final.append(sexo)

#Convierto de nuevo a tupla
tuple_modelo_final = tuple(l_tuple_modelo_final)

#Guardamos para el modelo final
lista_modelo += [tuple_modelo_final]




##GUARDAMOS A TXT LA LISTA_MODELO
df = pd.DataFrame(lista_modelo)

df.to_csv('lista_modelo.txt', index=False)






print('It took {0:0.2f} seconds'.format(time.time() - start_time))
