# Including needed libraries
library(qdapDictionaries)
library(qdapRegex)
library(qdapTools)
library(RColorBrewer)
library(qdap)
library(XML)
library(NLP)
library(tm)
library(splitstackshape)
library(lattice)
library(ggplot2)
library(caret)

#library(bitops)
#library(RCurl)

start.time <- Sys.time()

# Preparing parameters
n <- 50
lang <- "es"
path_training <- "C:/Users/DPEREZ/OneDrive/Master/15.-Text_Mining_Social_Media/pan-ap17-bigdata/training"
path_test <- "C:/Users/DPEREZ/OneDrive/Master/15.-Text_Mining_Social_Media/pan-ap17-bigdata/test"
k <- 3
r <- 1

# Auxiliar functions
# * GenerateVocabulary: Given a corpus (training set), obtains the n most frequent words
# * GenerateBoW: Given a corpus (training or test), and a vocabulary, obtains the bow representation

# GenerateVocabulary: Given a corpus (training set), obtains the n most frequent words
GenerateVocabulary <- function(path, n = 1000, lowcase = TRUE, punctuations = TRUE, numbers = TRUE, whitespaces = TRUE, swlang = "", swlist = "", verbose = TRUE, sex = "",pais = "") 
  {
  setwd(path)
  
  # Reading corpus list of files
  files = list.files(pattern="*.xml")
  
  
  tr = read.table("C:/Users/DPEREZ/OneDrive/Master/15.-Text_Mining_Social_Media/pan-ap17-bigdata/training/truth.txt", sep=";", encoding="UTF-8")
  #tr = tr[,c(1,4,7)]
  colnames(tr) <- c("id","sexo","pais")
  tr$id <- as.character(tr$id)
  
  
  # Reading files contents and concatenating into the corpus.raw variable
  corpus.raw <- NULL
  i <- 0
  for (file in files) 
  {
    # Para generar las palabras por sexo
    if(sex!="")
    {
      file2 <- gsub(".xml","",file)
      id_sex <- tr[which(tr$id == file2),2]
      
      if(id_sex == sex)
      {
        xmlfile <- xmlTreeParse(file, useInternalNodes = TRUE)
        corpus.raw <- c(corpus.raw, xpathApply(xmlfile, "//document", function(x) xmlValue(x)))
        i <- i + 1
        if (verbose) print(paste(i, " ", file))
      }
    }
    
    # Para generar las palabras por pais
    if(pais!="")
    {
      file2 <- gsub(".xml","",file)
      id_pais <- tr[which(tr$id == file2),3] 
      
      if(id_pais == pais)
      {
        xmlfile <- xmlTreeParse(file, useInternalNodes = TRUE)
        corpus.raw <- c(corpus.raw, xpathApply(xmlfile, "//document", function(x) xmlValue(x)))
        i <- i + 1
        if (verbose) print(paste(i, " ", file))
      }
      
    }
  }
  
  # Preprocessing the corpus
  corpus.preprocessed <- corpus.raw
  
  
  if (lowcase) {
    if (verbose) print("Tolower...")
    corpus.preprocessed <- tolower(corpus.preprocessed)
  }
  
  if (punctuations) {
    if (verbose) print("Removing punctuations...")
    corpus.preprocessed <- removePunctuation(corpus.preprocessed)
  }
  
  if (numbers) {
    if (verbose) print("Removing numbers...")
    corpus.preprocessed <- removeNumbers(corpus.preprocessed)
  }
  
  if (whitespaces) {
    if (verbose) print("Stripping whitestpaces...")
    corpus.preprocessed <- stripWhitespace(corpus.preprocessed)
  }
  
  if (swlang!="")	{
    if (verbose) print(paste("Removing stopwords for language ", swlang , "..."))
    corpus.preprocessed <- removeWords(corpus.preprocessed, stopwords(swlang))
  }
  
  if (swlist!="") {
    if (verbose) print("Removing provided stopwords...")
    corpus.preprocessed <- removeWords(corpus.preprocessed, swlist)
  }
  
  # Generating the vocabulary as the n most frequent terms
  if (verbose) print("Generating frequency terms")
  corpus.frequentterms <- freq_terms(corpus.preprocessed, n)
  
  if (verbose) plot(corpus.frequentterms)
  
  return (corpus.frequentterms)
}


# GENERACION DE VOCABULARIO
# GENERACION POR SEXO
vocabulary_male <- GenerateVocabulary(path_training, n, swlang=lang, sex="male")
vocabulary_female <- GenerateVocabulary(path_training, n, swlang=lang, sex="female")

connection_female<-file("../female.txt",encoding="UTF-8")
connection_male<-file("../male.csv",encoding="UTF-8")
write.csv(vocabulary_female,connection_female)
write.table(vocabulary_male,connection_male)


# GENERACION POR PAIS
vocabulary_Argentina <- GenerateVocabulary(path_training, n, swlang=lang, pais="argentina")
vocabulary_Chile <- GenerateVocabulary(path_training, n, swlang=lang, pais="chile")
vocabulary_Colombia <- GenerateVocabulary(path_training, n, swlang=lang, pais="colombia")
vocabulary_Mexico <- GenerateVocabulary(path_training, n, swlang=lang, pais="mexico")
vocabulary_Peru <- GenerateVocabulary(path_training, n, swlang=lang, pais="peru")
vocabulary_Spain <- GenerateVocabulary(path_training, n, swlang=lang, pais="spain")
vocabulary_Venezuela <- GenerateVocabulary(path_training, n, swlang=lang, pais="venezuela")

wite.csv(vocabulary_Argentina,"argentina.csv")
wite.csv(vocabulary_Chile,"chile.csv")
wite.csv(vocabulary_Colombia,"colombia.csv")
wite.csv(vocabulary_Mexico,"mexico.csv")
wite.csv(vocabulary_Peru,"peru.csv")
wite.csv(vocabulary_Spain,"spain.csv")
wite.csv(vocabulary_Venezuela,"podemos.csv")




