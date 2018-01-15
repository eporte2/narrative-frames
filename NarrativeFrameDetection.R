#NARRATIVE FRAME BOUNDARY DETECTION
#By Eva Portelance
#June 1, 2017
#
#
#Input: set of txt files (Novels)
#Output: a csv for each txt file with list of narrative frame boundary locations
#
#
#STEP 1: load dependencies
library("tm")
library("SnowballC")
library("RWeka")
library("NLP")
library("openNLP")
library("openNLPdata")
library("proxy")
#
#
#
#STEP 2: define directory containing txt files
dir <- "Clean_texts"
#
#Note: this script was written on mac and uses '/' in directory names. If you are on a pc, you may need to change the directions to '\'
#for certain lines in script.
#
#
#
#
#STEP 3: define your variables (defaults are what we found to be ideal)
#size of windows of text to be compared
window <- 1000
#size of step in words for every itteration
step <- 100
#
#
#
#STEP 4: run all of the following code!!!
#****************************************#

## AUXILIARY FUNCTIONS FOR NNP and NN_P MODELS

#Clean Text of excess whitespaces parameter1: String (text) returns: String 
#Must keep all punctuation, capital letters and morphology for POS annotation
cleanText <- function(text) {
  text_clean <- stripWhitespace(text) # Reduce all white spaces to 1
  text_whole <- paste(text_clean, collapse=" ") # collapse into single chunk
  text_char <- as.String(text_whole)
  
  return(text_char)
}


#POS annotation of text parameter1: String returns: Annotated string
annotatePOS <- function(text_char) {
  #sentence and word annotation necessary for POS annotation
  sent_token_annotator <- Maxent_Sent_Token_Annotator(language = "en")
  word_token_annotator <- Maxent_Word_Token_Annotator(language = "en")
  #Sent-token-an and word-token-an pipeline as word-an requires sent-an input
  an1 <- annotate(text_char, list(sent_token_annotator, word_token_annotator))
  
  #Annotate POS, uses Penn Treebank Project annotation scheme : http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
  pos_tag_annotator <- Maxent_POS_Tag_Annotator(language = "en")
  an2 <- annotate(text_char, pos_tag_annotator, an1)
  
  #Determine POS tag for word token distribution
  an3w <- subset(an2, type == "word")
  tags <- sapply(an3w$features, `[[`, "POS")
  
  #concatenate annotated word_POS into single chunk
  text_tagged <- paste(sprintf("%s_%s", text_char[an3w], tags), collapse=" ")
  
  return(text_tagged)
}


#Create Word vector of word_TAG parameter1: String (text) Returns Word Vector (String[])
splitText <- function(text_tagged) {
  return (unlist (strsplit (text_tagged, " ")))
}


#Selecting only the words of specified POS category parameter1: String array ie word vector (of format word_TAG)
# parameter2: Regex String (selected category for comparison form AAA), returns: word vector of specified tag
selectPOS <- function(words_tagged, target_tag) {
  return (words_tagged[grep(target_tag, words_tagged)] ) 
}


#Removing Tags from words parameter1: word vector returns: word vector
removeTags <- function(words_tag) {
  return (gsub("_.*$", "", words_tag))
}

#CRUCIAL STEP: MODEL = Select POS for comparison depending on hypothesis variable (Proper Nouns, Verbs (Tense), Adverbs, other contentful Nouns... )
#See http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html for TAG definitions
modelTest <- function(text,model,mod,window,step) {
  chunks <- step*(round(length(text)/step))
  #instantiate data frame for comparison results
  cos.df<-NULL
  #Loop through 2*1000 word chunks, shifting by 100 each loop
  for(i in seq(from=1, to=(chunks-(2*window)), by=step)){
    #Create 2 sub parts of 1000 words for comparison
    part1_sub <- text[i:(i+window-1)]
    part2_sub <- text[(i+window):(i+(2*window-1))]
    #Concatenate into String object
    part1_char <- cleanText(part1_sub)
    part2_char <- cleanText(part2_sub)
    
    ###FIRST STEP: Create Local Dictionary
    #Annotate POS 
    part1_tag <- annotatePOS(part1_char)
    part2_tag <- annotatePOS(part2_char)
    #Transform annotated text into vector
    words1_vec <- splitText(part1_tag)
    words2_vec <- splitText(part2_tag)
    if(mod==1){
      #Filter word list by model
      words1_tag <- selectPOS(words1_vec, model)
      words2_tag <- selectPOS(words2_vec, model)
      #Remove tags from selected words
      words1 <- removeTags(words1_tag)
      words2 <- removeTags(words2_tag)
    }
    else{
      words1 <- removeTags(words1_vec)
      words2 <- removeTags(words2_vec)
    }
    #All to lower Case
    words1 <- tolower(words1)
    words2 <- tolower(words2)
    #Remove all duplicates from dictionary
    V1 <- unique(c(words1, words2))
    words_unique <- as.data.frame(V1,stringsAsFactors=FALSE)
    if(length(V1) != 0){
      #SECOND STEP: Comparison of dictionary word frequency
      part1_vs <- VectorSource(c(part1_char))
      part2_vs <- VectorSource(c(part2_char))
      #Create Tokenizer, now set at (1,1) : 1 word = 1 token, but can play with this if we decide to search for Phrase groups, ex NP, VP.. This also requires
      #playing with the RegEx Select POS for dictionary
      NgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1)) #min/max = ngram
      options(mc.cores=1) #this makes sure that the number of processes running at the same time is 1 (not to overload processors...computer hardware stuff)
      #Create and clean corpus1
      corpus1 <- VCorpus(part1_vs, readerControl=list(language="English"))
      corpus1 <- tm_map(corpus1, content_transformer(stripWhitespace))
      corpus1 <- tm_map(corpus1, content_transformer(tolower))
      corpus1 <- tm_map(corpus1, content_transformer(removePunctuation))
      corpus1 <- tm_map(corpus1, content_transformer(removeNumbers))  
      #Create TDMs - TDMs are matrices that describe Term frequency
      corpus1_tdm<-TermDocumentMatrix(corpus1, control = list(tokenize = NgramTokenizer))
      #Create a column with all words to adjoin to tdm
      tdm1_matrix <- as.matrix(corpus1_tdm)
      #Create and clean corpus2
      corpus2 <- VCorpus(part2_vs, readerControl=list(language="English"))
      corpus2 <- tm_map(corpus2, content_transformer(stripWhitespace))
      corpus2 <- tm_map(corpus2, content_transformer(tolower))
      corpus2 <- tm_map(corpus2, content_transformer(removePunctuation))
      corpus2 <- tm_map(corpus2, content_transformer(removeNumbers))
      corpus2_tdm<-TermDocumentMatrix(corpus2, control = list(tokenize = NgramTokenizer))
      tdm2_matrix <- as.matrix(corpus2_tdm)
      
      #DATA FRAME PART 1:
      V1<-row.names(tdm1_matrix)
      tdm1_matrix<-cbind(V1, tdm1_matrix)
      #Data Frame Part1 : Word-counts
      # Match Dictionary words to words in Column 1 (i.e. part 1 tokenized words) -NOTE: I am not sure why the if-else sequence is necessary:
      # -> Why differenciate cases of length 1 or 2 ?
      if (length(tdm1_matrix[V1 %in% words_unique$V1,]) == 0) {
        tdm1_sub<-tdm1_matrix[V1 %in% words_unique$V1,]
      } else if (length(tdm1_matrix[V1 %in% words_unique$V1,]) <= 2) {
        tdm1_sub<-tdm1_matrix[V1 %in% words_unique$V1,]
        tdm1_sub<-data.frame(tdm1_sub[[1]], tdm1_sub[[2]])
      } else { 
        tdm1_sub<-tdm1_matrix[V1 %in% words_unique$V1,]
      }
      #If we manually want to look at most frequent words for testing, use this line to order frequencies
      #tdm1_sort<-tdm1_sub[order(-as.numeric(tdm1_sub[,2])),]
      
      #Give 0 count to all words that did not match
      non1_wds_unique<-words_unique[!words_unique$V1 %in% tdm1_sub[,1],]
      if(length(non1_wds_unique) != 0){
        non1_df<-cbind(non1_wds_unique, 0)
        colnames(non1_df)<-c("word", "count")
        colnames(tdm1_sub)<-c("word", "count")
        #Combine the two df to have one df for all unique word counts from part1
        wordcount1_df<-rbind(tdm1_sub, non1_df)
      }
      else{
        colnames(tdm1_sub)<-c("word", "count")
        #Combine the two df to have one df for all unique word counts from part1
        wordcount1_df<-tdm1_sub
      }
      wordcount.source<-wordcount1_df
      
      #DATA FRAME PART 2
      V2<-row.names(tdm2_matrix)
      tdm2_matrix<-cbind(V2, tdm2_matrix)
      #Data Frame Part2 : Word-counts
      # Match Dictionary words to words in Column 2 (i.e. part 2 tokenized words) -NOTE: I am not sure why the if-else sequence is necessary:
      # -> Why differenciate cases of length 1 or 2 ?
      if (length(tdm2_matrix[V2 %in% words_unique$V1,]) == 0) {
        tdm2_sub<-tdm2_matrix[V2 %in% words_unique$V1,]
      } else if (length(tdm2_matrix[V2 %in% words_unique$V1,]) <= 2) {
        tdm2_sub<-tdm2_matrix[V2 %in% words_unique$V1,]
        tdm2_sub<-data.frame(tdm2_sub[[1]], tdm2_sub[[2]])
      } else { 
        tdm2_sub<-tdm2_matrix[V2 %in% words_unique$V1,]
      }
      #If we manually want to look at most frequent words for testing, use this line to order frequencies
      #tdm2_sort<-tdm2_sub[order(-as.numeric(tdm2_sub[,2])),]
      
      #Give 0 count to all words that did not match
      non2_wds_unique<-words_unique[!words_unique$V1 %in% tdm2_sub[,1],]
      if(length(non2_wds_unique) != 0){
        non2_df<-cbind(non2_wds_unique, 0)
        colnames(non2_df)<-c("word", "count")
        colnames(tdm2_sub)<-c("word", "count")
        #Combine the two df to have one df for all unique word counts from part2
        wordcount2_df<-rbind(tdm2_sub, non2_df)
      }
      else{
        colnames(tdm2_sub)<-c("word", "count")
        wordcount2_df<-tdm2_sub
      }
      wordcount.target<-wordcount2_df
      
      #MERGE TDMs
      wordcount_all<-merge(wordcount.source, wordcount.target, by = "word", all=TRUE)
      row.names(wordcount_all)<-wordcount_all[,1]
      wordcount_all<-wordcount_all[,-1]
      #This is the final word frequency matrix which contains counts for Parts 1 and 2 to look at
      wordcount_all<-t(wordcount_all)
      #Make sure that vectors are of numeric type before cosine similarity test
      source.v<-as.numeric(wordcount_all[1,])
      target.v<-as.numeric(wordcount_all[2,])
      #Make a test matrix without reference word column (2 columns of numbers only)
      test.matrix<-rbind(source.v, target.v)
      #FINALLY: Compare similarity using cosine sim. test (can play with this by using different similarity tests too)
      cosine.sim<-simil(test.matrix, method = "cosine")
      result<-cosine.sim[1]
      temp.df<-data.frame(i, result)
      #for inspecting stop here with temp.df
      #add result to comparison list
    }
    else {
      result<-0
      temp.df<-data.frame(i, result)
    }
    cos.df<-rbind(cos.df, temp.df)
  }
  return (cos.df)
}


#Filter cosine results and keep only those bellow cutoff
refineCos <- function(cos_df) {
  diff <- (t.test(cos_df[,2])$conf.int[1])#below lower 95% conf interval
  for(i in 1:(nrow(cos_df))) {
    if(cos_df[i,2] > diff) {
      cos_df[i,2] <- 0
    } 
  }
  return (cos_df)
}

##This filter selects all local minimums 
filterModel1 <- function(model.final, k){
  ##to filter out over predictions as our window for a correct prediction is 400 words
  i<-1
  while(i != (nrow(model.final)-1)){
    cos <- model.final[i,k]
    flw <- model.final[i+1,k]
    ## a row with a prediction , i.e one of them is not equal to zero
    if (cos != flw){
      if(cos != 0 && flw != 0){
        if(cos < flw){
          temp <- cos
          while(cos < flw && flw != 0 && i != (nrow(model.final)-2)){
            if(cos > temp){
              model.final[i,k] <- 0
            }
            i<-i+1
            cos <- model.final[i,k]
            flw <- model.final[i+1,k]
          }
          if(flw == 0 && i > 1){
            if(cos > temp){
              model.final[i,k] <- 0
            }
          }
        }
        if(cos > flw){
          model.final[i,k] <- 0
        }
      }
    }
    i<-i+1
  }
  return (model.final)
}

## ALL MODEL AUXILIAIRY FUNCTIONS

#Clean Text of excess whitespaces parameter1: String (text) returns: String 
cleanTextstem <- function(text) {
  text_clean <- stripWhitespace(text) # Reduce all white spaces to 1
  text_whole <- removePunctuation(text_clean)
  ## Remove stopwords, no stemming for best performance
  text_whole <- removeWords(text_whole, stopwords("English"))
  #text_whole<- stemDocument(text_whole, language="english")
  text_whole <- paste(text_whole, collapse=" ")# collapse into single chunk
  text_char <- as.String(text_whole)
  
  return(text_char)
}

modelTeststem <- function(text, window, step) {
  chunks <- step*(round(length(text)/step))
  #instantiate data frame for comparison results
  cos.df<-NULL
  #Loop through 2*1000 word chunks, shifting by 100 each loop
  for(i in seq(from=1, to=(chunks-(2*window)), by=step)){
    #Create 2 sub parts of 1000 words for comparison
    part1_sub <- text[i:(i+(window -1))]
    part2_sub <- text[(i+window):(i+(2*window)-1)]
    #Concatenate into String object
    part1_char <- cleanTextstem(part1_sub)
    part2_char <- cleanTextstem(part2_sub)
    
    ###FIRST STEP: Create Local Dictionary
    #Transform text into vector
    words1 <- splitText(part1_char)
    words2 <- splitText(part2_char)
    #All to lower Case
    words1 <- tolower(words1)
    words2 <- tolower(words2)
    #Remove all duplicates from dictionary
    V1 <- unique(c(words1, words2))
    words_unique <- as.data.frame(V1,stringsAsFactors=FALSE)
    if(length(V1) != 0){
      #SECOND STEP: Comparison of dictionary word frequency
      part1_vs <- VectorSource(c(part1_char))
      part2_vs <- VectorSource(c(part2_char))
      #Create Tokenizer, now set at (1,1) : 1 word = 1 token, but can play with this if we decide to search for Phrase groups, ex NP, VP.. This also requires
      #playing with the RegEx Select POS for dictionary
      NgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1)) #min/max = ngram
      options(mc.cores=1) #this makes sure that the number of processes running at the same time is 1 (not to overload processors...computer hardware stuff)
      #Create and clean corpus1
      corpus1 <- VCorpus(part1_vs, readerControl=list(language="English"))
      corpus1 <- tm_map(corpus1, content_transformer(stripWhitespace))
      corpus1 <- tm_map(corpus1, content_transformer(tolower))
      corpus1 <- tm_map(corpus1, content_transformer(removePunctuation))
      corpus1 <- tm_map(corpus1, content_transformer(removeNumbers)) 
      corpus1 <- tm_map(corpus1, removeWords, stopwords("English"))
      #corpus1 <- tm_map(corpus1, stemDocument, language = "english")
      #Create TDMs - TDMs are matrices that describe Term frequency
      corpus1_tdm<-TermDocumentMatrix(corpus1, control = list(tokenize = NgramTokenizer))
      #Create a column with all words to adjoin to tdm
      tdm1_matrix <- as.matrix(corpus1_tdm)
      #Create and clean corpus2
      corpus2 <- VCorpus(part2_vs, readerControl=list(language="English"))
      corpus2 <- tm_map(corpus2, content_transformer(stripWhitespace))
      corpus2 <- tm_map(corpus2, content_transformer(tolower))
      corpus2 <- tm_map(corpus2, content_transformer(removePunctuation))
      corpus2 <- tm_map(corpus2, content_transformer(removeNumbers))
      corpus2 <- tm_map(corpus2, removeWords, stopwords("English"))
      #corpus2 <- tm_map(corpus2, stemDocument, language = "english")
      corpus2_tdm<-TermDocumentMatrix(corpus2, control = list(tokenize = NgramTokenizer))
      tdm2_matrix <- as.matrix(corpus2_tdm)
      
      #DATA FRAME PART 1:
      V1<-row.names(tdm1_matrix)
      tdm1_matrix<-cbind(V1, tdm1_matrix)
      #Data Frame Part1 : Word-counts
      # Match Dictionary words to words in Column 1 (i.e. part 1 tokenized words) -NOTE: I am not sure why the if-else sequence is necessary:
      # -> Why differenciate cases of length 1 or 2 ?
      if (length(tdm1_matrix[V1 %in% words_unique$V1,]) == 0) {
        tdm1_sub<-tdm1_matrix[V1 %in% words_unique$V1,]
      } else if (length(tdm1_matrix[V1 %in% words_unique$V1,]) <= 2) {
        tdm1_sub<-tdm1_matrix[V1 %in% words_unique$V1,]
        tdm1_sub<-data.frame(tdm1_sub[[1]], tdm1_sub[[2]])
      } else { 
        tdm1_sub<-tdm1_matrix[V1 %in% words_unique$V1,]
      }
      #If we manually want to look at most frequent words for testing, use this line to order frequencies
      #tdm1_sort<-tdm1_sub[order(-as.numeric(tdm1_sub[,2])),]
      
      #Give 0 count to all words that did not match
      non1_wds_unique<-words_unique[!words_unique$V1 %in% tdm1_sub[,1],]
      if(length(non1_wds_unique) != 0){
        non1_df<-cbind(non1_wds_unique, 0)
        colnames(non1_df)<-c("word", "count")
        colnames(tdm1_sub)<-c("word", "count")
        #Combine the two df to have one df for all unique word counts from part1
        wordcount1_df<-rbind(tdm1_sub, non1_df)
      }
      else{
        colnames(tdm1_sub)<-c("word", "count")
        #Combine the two df to have one df for all unique word counts from part1
        wordcount1_df<-tdm1_sub
      }
      wordcount.source<-wordcount1_df
      
      #DATA FRAME PART 2
      V2<-row.names(tdm2_matrix)
      tdm2_matrix<-cbind(V2, tdm2_matrix)
      #Data Frame Part2 : Word-counts
      # Match Dictionary words to words in Column 2 (i.e. part 2 tokenized words) -NOTE: I am not sure why the if-else sequence is necessary:
      # -> Why differenciate cases of length 1 or 2 ?
      if (length(tdm2_matrix[V2 %in% words_unique$V1,]) == 0) {
        tdm2_sub<-tdm2_matrix[V2 %in% words_unique$V1,]
      } else if (length(tdm2_matrix[V2 %in% words_unique$V1,]) <= 2) {
        tdm2_sub<-tdm2_matrix[V2 %in% words_unique$V1,]
        tdm2_sub<-data.frame(tdm2_sub[[1]], tdm2_sub[[2]])
      } else { 
        tdm2_sub<-tdm2_matrix[V2 %in% words_unique$V1,]
      }
      #If we manually want to look at most frequent words for testing, use this line to order frequencies
      #tdm2_sort<-tdm2_sub[order(-as.numeric(tdm2_sub[,2])),]
      
      #Give 0 count to all words that did not match
      non2_wds_unique<-words_unique[!words_unique$V1 %in% tdm2_sub[,1],]
      if(length(non2_wds_unique) != 0){
        non2_df<-cbind(non2_wds_unique, 0)
        colnames(non2_df)<-c("word", "count")
        colnames(tdm2_sub)<-c("word", "count")
        #Combine the two df to have one df for all unique word counts from part2
        wordcount2_df<-rbind(tdm2_sub, non2_df)
      }
      else{
        colnames(tdm2_sub)<-c("word", "count")
        wordcount2_df<-tdm2_sub
      }
      wordcount.target<-wordcount2_df
      
      #MERGE TDMs
      wordcount_all<-merge(wordcount.source, wordcount.target, by = "word", all=TRUE)
      row.names(wordcount_all)<-wordcount_all[,1]
      wordcount_all<-wordcount_all[,-1]
      #This is the final word frequency matrix which contains counts for Parts 1 and 2 to look at
      wordcount_all<-t(wordcount_all)
      #Make sure that vectors are of numeric type before cosine similarity test
      source.v<-as.numeric(wordcount_all[1,])
      target.v<-as.numeric(wordcount_all[2,])
      #Make a test matrix without reference word column (2 columns of numbers only)
      test.matrix<-rbind(source.v, target.v)
      #FINALLY: Compare similarity using cosine sim. test (can play with this by using different similarity tests too)
      cosine.sim<-simil(test.matrix, method = "cosine")
      result<-cosine.sim[1]
      temp.df<-data.frame(i, result)
      #for inspecting stop here with temp.df
      #add result to comparison list
    }
    else {
      result<-0
      temp.df<-data.frame(i, result)
    }
    cos.df<-rbind(cos.df, temp.df)
  }
  return (cos.df)
}

#depth result data frame calculator. Helper for filterModel2
depth<-function(model.final,k){
  depths.rec<-NULL
  for(i in 1:(nrow(model.final))){
    prv <- i
    flw <- i
    delta1<-0
    delta2<-0
    if(model.final[i,k] > 0){
      while(prv != 1){
        cos<-model.final[prv,k]
        prv<-prv-1
        if(cos<=model.final[prv,k]){
          cos<-model.final[prv,k]
        }
        else{
          delta1<-cos-model.final[i,k]
          break
        } 
      }
      while(flw < (nrow(model.final)-1)){
        cos<-model.final[flw,k]
        flw<-flw+1
        if(cos<=model.final[flw,k]){
          cos<-model.final[flw,k]
        }
        else{
          delta2<-cos-model.final[i,k]
          break
        }
      }
    }
    depth<-delta1+delta2
    temp<-data.frame(i,depth)
    depths.rec <-rbind(depths.rec,temp)
  }
  return(depths.rec)
}

Cutoff<-function(model.final, k){
  model_no0<-NULL
  for(i in 1:nrow(model.final)){
    if(model.final[i,k]>0){
      m<-model.final[i,k]
      temp<-data.frame(i,m)
      model_no0<-rbind(model_no0,temp)
    }
  }
  cutoff<-(t.test(model_no0[,2])$conf.int[1])
  #cutoff<-(mean(model_no0[,2]))
  #cutoff<- mean(model_no0[,2])-sd(model_no0[,2])/2  #Hearst's cutoff
  return (cutoff)
}

##This filter uses the most important depth scores (same scheme as Hearth 1994)
filterModel2 <-function(model.final, k){
  depths<-depth(model.final,k)
  i<-1
  cutoff <-Cutoff(depths,2)
  while(i <= (nrow(depths)-1)){
    prv <- depths[i,2]
    flw <- depths[i+1,2]
    if(prv<=flw){
      depths[i,2]<-0  
    }
    else{
      i<-i+1
      while(flw != 0 && i<(nrow(depths)-1)){
        depths[i,2]<-0
        i<-i+1
        flw<-depths[i,2]
      }
    }
    i<-i+1
  }
  for(j in 1:(nrow(depths))){
    if(depths[j,2] >= cutoff){
      depths[j,2]<-0
    }
  }
  return (depths)
}

## Combine model predictions
## NNP + ALL + NN_P
combinationModel4 <- function(model.final) {
  comb<-NULL
  for(i in 1:nrow(model.final)){
    COMB<-((i-1)*100)+1000
    #prioratise model NNP
    if(model.final[i,3] != 0){
      num <-model.final[i,3]
      temp<-data.frame(COMB,num)
    }
    else {
      if(model.final[i,4] != 0){
        num <-model.final[i,4]
        temp<-data.frame(COMB,num)
      }
      else {
        if(model.final[i,2] != 0){
          num <-model.final[i,2]
          temp<-data.frame(COMB,num)
        }
        else{
          num <-0
          temp<-data.frame(COMB,num)
        }
      }
    }
    comb<-rbind(comb,temp)
  }
  model.final<-cbind(model.final,comb)
  return (model.final)
}

### MAIN METHOD
files <- list.files(path=dir, pattern="*.txt", full.names=T, recursive=FALSE)
lapply(files, function(x) {
  text <- scan(x, what = "character", quote = "", encoding = "UTF-8")
  #model NN_P all contentful nouns
  model1.c <- refineCos(modelTest(text,"(NN)[^(NNP)]|(NN)$",1, window, step))
  NN_P<-model1.c[,1]+999
  model1.fnl <-cbind(NN_P,model1.c[,2])
  model1.fnl <-filterModel1(model1.fnl,2)
  #model2 NNP all proper nouns
  model2.c <- refineCos(modelTest(text,"(NNP)",1, window, step))
  model2.fnl <-filterModel1(model2.c,2)
  #model3 ALL (all words - stop words, no stemming)
  model3.c <- modelTeststem(text, window, step)
  model3.fnl <-filterModel2(model3.c,2)
  model.final <- cbind(model1.fnl,model2.fnl[,2],model3.fnl[,2])
  #combine results
  model.final <- combinationModel4(model.final)
  model.final <- filterModel1(model.final, 6)   
  locations <- NULL
  for( i in 1:nrow(model.final)){
    if(model.final[i,6] != 0){
      locations <- rbind(locations,model.final[i,5])
    }
  }
  write.csv(locations, file=paste(x ,".locations.csv", sep = ""))
})




