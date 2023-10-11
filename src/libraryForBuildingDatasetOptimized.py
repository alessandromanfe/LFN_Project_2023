''' LIBRARY of functions defined by us to pre-process and encode source articles and highlights sentences '''

# to use it as module in Google Colab
'''
import sys
sys.path.insert(0, '/content/')
import libraryForBuildingDatasetOptimized as l4bdOptimized
'''

# required external libraries
'''
!pip install transformers
!pip install datasets
!pip install contractions
!pip install nltk

!conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch # depends on GPU/CPU
'''

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd

from datasets import load_dataset
import contractions
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


''' Loading functions '''


# load dataset from huggingface repository
def loadCnnDaylyMailDataset():
    return load_dataset('cnn_dailymail', '3.0.0')


# load pretrained BERT tokenizer, note the model is CASE sensitive (if not the model particular subtokenization could not allow to retrive which word were Capital/proper nouns)
def loadPretrainedBERTmodelTokenizer(BERT_MODEL = 'bert-base-cased'):
    return BertTokenizer.from_pretrained(BERT_MODEL)
    
    
# load pretrained BERT model
def loadPretrainedBERTmodel(BERT_MODEL = 'bert-base-cased'):
    model = BertModel.from_pretrained(BERT_MODEL, output_hidden_states = True, ) # we retrieve and use all hidden states
    model.eval() # put it in inference mode
    return model


# load final vocabulary using topKEnglishWords file and add missing characters, digits and tokens
def loadTargetVocabulary(numberOfTopkEnglishWords, numberOfUnkTags, path='/content/topEnglishWords.csv'):
    vocabulary_df = pd.read_csv(path)
    wordsList30k = vocabulary_df[0:numberOfTopkEnglishWords].values[:,0].tolist() # 30k top frequent english words
    # checking that digits and specials charachters are in there, and also [EOS]
    digitsAndSpecChars = ['!', '"', '£', '$', '€', '%', '&', '/', '(', ')', '[', ']', '{', '}', '?', '.', ',', ';', ':', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '--', "''", "``"]
    for item in digitsAndSpecChars:
      if item not in wordsList30k:
        wordsList30k.append(item)
    # tags for UNKNOWN words (NUMBER OF UNKNOWN WORDS 100)
    for i in range(numberOfUnkTags):
      unknownTag = '[UNK' + str(i+1) + ']'
      wordsList30k.append(unknownTag)
    # adding [SEP] to define end of sentence but followed by another one (in BERT EMBEDDING SEP indicates general sentence ending)
    # [EOS] to indicate End of Summary (to STOP the model outputting words)
    # [UNK4BOTHDICT] if the word does not belong neither to final vocabulary nor to UNKTags dictionary from source article
    wordsList30k.append('[UNK4BOTHDICT]')
    wordsList30k.append('[SEP]')
    wordsList30k.append('[EOS]')
    wordsList30k.append('[START]') # TO BE USED AS FIRST TOKEN IN SUMMARY !?
    wordsList30k.append('[PAD]')

    # Swapping [PAD] with first word
    temp = wordsList30k[0]
    wordsList30k[0] = wordsList30k[-1]
    wordsList30k[-1] = temp

    # words as values in a dictionary for faster search
    finalVocabularyAsDict = {x : '' for x in wordsList30k}
    return finalVocabularyAsDict
    
    
# define an embedding for each UNK tag (768 zeros followed by 2 values ranging [5 to 5 - 0.1 *numberOfUnkTags]
def defineDictionaryOfEmbeddingsForUnknownTAGS(numberOfUnkTags):
    dictionaryOfEmbeddingsForUnknownTAGS = {}
    firstToLastValue = 5
    lastValue = 5
    unkTagBaseEmbedding = torch.zeros(770)
    unkTagBaseEmbedding[-2] += firstToLastValue
    unkTagBaseEmbedding[-1] += lastValue

    for i in range(numberOfUnkTags):
      unknownTag = '[UNK' + str(i+1) + ']'
      dictionaryOfEmbeddingsForUnknownTAGS[unknownTag] = unkTagBaseEmbedding
      unkTagBaseEmbedding[-2] -= 0.1
      unkTagBaseEmbedding[-1] -= 0.1
    return dictionaryOfEmbeddingsForUnknownTAGS



''' ---------------------------- HELPER FUNCTIONS to construct dataset ---------------------------- '''



''' Given a text/article/sentence expands the contractions in it eg. I'll --> I will '''
def expandContractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    return expanded_text


''' Split sentences in a text '''
def splitSentences(text):
    return sent_tokenize(text)


'''
Preprocess a single sentence and returns (list of WORDS, wordLevelEmbeddingsAsTensor)
The list of words contains words from original sentence eg "dog" or [UNK1] tags if recognized as OOV word
wordLevelEmbeddingsAsTensor has shape (#wordsInSentence, #770)

NB: position encoding for transformer and padding to max length must be done in first Transformer layer

uses unknownWordsDictionaryForAnArticlen from a previous sentence within the SAME article to encode already seen OOV words
using same UNK tag number

needs finalVocabularyAsDict to determine if a word is OOV even for final Vocabulary

'''
def preprocessSingleSentence(sentence, tokenizer, model, unknownWordsDictionaryForAnArticle, 
          dictionaryOfEmbeddingsForUnknownTAGS, finalVocabularyAsDict):
          
    # we discard the article if there are too many UNK words (more than len(dictionaryOfEmbeddingsForUnknownTAGS)
    discardTheArticle = False
    
    # add [CLS] and [SEP] tags
    marked_sentence = "[CLS] " + sentence[:-1] + ' ' + sentence[-1][:-1] + " [SEP]" # removing '.' and adding [SEP] to mark end of sentence as BERT requires
    # tokenize sentence using BERT specific tokenizer
    tokenized_sentence = tokenizer.tokenize(marked_sentence) # here just word level tokenization
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence) # subword level tokenization as BERT does
    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_sentence)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    # without constructing computational graph
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2] # retrieve all hidden states outputs
    
    # Concatenate the tensors for all layers. We use `stack` here to create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    
    # we now sum the vectors from the last four layers as many researchers suggest
    token_vecs_sum = []
    for token in token_embeddings:
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    
    # merging tokens to obtain word level lists
    # eg L, ##ON, ##DON will be merged into LONDON, the 2 of the 3 embeddings for each token are discarded (this embedding will be later modified)
    
    wordsListWithTAGS = [] # to append only when word is completed and converted to UNKTAG if needed
    wordLevelEmbedList = [] # to append only when word is completed and embedding is produces
    tempWord = ''
    positionStarting = 0
    for i, elem in enumerate(zip(tokenized_sentence, token_vecs_sum)):
      token, embedding = elem
      # if starts another token without ## prefix we append previous word now finished
      if token[0] != '#' and tempWord != '':
        # but if digit we would like to continue concatenating
        if token[0].isdigit() and ((tempWord[0] in ['€', '£', '$', '%']) or tempWord[-1].isdigit() or tempWord[-1] == '.'):
          tempWord += token
          # eg. 41 . 1 we would like to concatenate the dot with previous tokens
        elif token[0] == '.' and i < len(tokenized_sentence)-1 and tokenized_sentence[i+1] != ' ':
          tempWord += token
        else:
          # wordsListWithTAGS.append(tempWord) # for debugging from token to word
          # WORKING ON tempWord (that is now considered finished and to be added to list and produced embedding)
          # create TAG if needed:
          # if the word starts by a special symbol like $ or digit (we treat it as UNK)
          isUNK = False
          if tempWord[0] in ['$', '€', '£', '%'] or tempWord[0].isdigit():
            isUNK = True
          elif tempWord[0].isupper():
            # is upper and belongs to final vocabulary and is after a dot or at beginning of sentence (after [CLS]) eg ". He"
            if len(wordsListWithTAGS) != 0 and wordsListWithTAGS[-1] in ['.', '!', '?', '[CLS]'] and tempWord.lower() in finalVocabularyAsDict:
              isUNK = False
            else:
              isUNK = True
          else:
              isUNK = False

          # if isUNK we retrieve or generate TAG and finally append to wordsListWithTAGS
          if isUNK == True:
            # if is UNK we check if already has a UNK tag number within that article dictionary, otherwise we create it
            if tempWord not in unknownWordsDictionaryForAnArticle:
              if len(unknownWordsDictionaryForAnArticle) < len(dictionaryOfEmbeddingsForUnknownTAGS):
                unknownWordsDictionaryForAnArticle[tempWord] = '[UNK' + str(len(unknownWordsDictionaryForAnArticle)+1) + ']'
                wordsListWithTAGS.append('[UNK' + str(len(unknownWordsDictionaryForAnArticle)) + ']')
              else:
                wordsListWithTAGS.append('[UNK' + '90' + ']')
                discardTheArticle = True
                return None, None, discardTheArticle
            else:
              unknownTagFromWordsInDict = unknownWordsDictionaryForAnArticle[tempWord]
              wordsListWithTAGS.append(unknownTagFromWordsInDict)
          # else is not UNK and we append original tempWord
          else:
            wordsListWithTAGS.append(tempWord)

          # We have now to generate embedding for TAG or retrieve and modify(adding 00) existing embedding
          word = wordsListWithTAGS[-1] #we just inserted it
          if isUNK == True:
            if word in dictionaryOfEmbeddingsForUnknownTAGS:
              newEmbedding = dictionaryOfEmbeddingsForUnknownTAGS[word]
            else:
              newEmbedding = dictionaryOfEmbeddingsForUnknownTAGS['[UNK90]']
              print("Number of OOV words greater than UNK tags")
          else:
            newEmbedding = torch.cat((token_vecs_sum[positionStarting], torch.zeros(2)))
                
          wordLevelEmbedList.append(newEmbedding)

          # we added the previous completed word, we now use current token as starting for potential new word
          tempWord = token
          positionStarting = i

      # if not subtoken we start appending to empty sentence
      elif token[0] != '#' and tempWord == '':
        tempWord += token
      # if subtoken we append BUT removing ## prefix
      else:
        tempWord += token[2:]

    # we missed out last token that is always SEP
    wordsListWithTAGS.append('[SEP]')
    wordLevelEmbedList.append(torch.cat((token_vecs_sum[-1], torch.zeros(2))))

    wordLevelEmbeddingsAsTensor = torch.stack(wordLevelEmbedList)[1:] # converting to tensor and removing CLS first token
    wordsListWithTAGS = wordsListWithTAGS[1:]
    
    if len(wordsListWithTAGS) != wordLevelEmbeddingsAsTensor.shape[0]:
        raise Exception("INCONSISTENCY IN LENGTH WHEN PRODUCING EMBEDDING FOR INPUT SENTENCE")
    
    return wordsListWithTAGS, wordLevelEmbeddingsAsTensor.type(torch.float16), discardTheArticle
    
    
''' Given an article, splits the sentences, produces an embedding for each sentence with a common OOV words dictionary and
    returns:
    - list of wordsListWithTAGS (1 for each sentence)
    - list of Tensors (#wordsEachSentence, 770)
    - unknownWordsDictionaryForAnArticle for the article
    - True if the article is to be discarded (too many UNK words)
'''
def preprocessArticle(article, tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS):
    article = expandContractions(article)
    articleSentences = splitSentences(article)
    
    unknownWordsDictionaryForAnArticle = {} # a dictionary for keeping track of UNK assigments FOR EACH ARTICLE
    
    wordsListWithTAGSList = []
    wordLevelEmbeddingsAsTensorList = []
    for sentence in articleSentences:
        wordsListWithTAGS, wordLevelEmbeddingsAsTensor, discardTheArticle = preprocessSingleSentence(sentence,
                                                                                  tokenizer,  
                                                                                  model, 
                                                                                  unknownWordsDictionaryForAnArticle, 
                                                                                  dictionaryOfEmbeddingsForUnknownTAGS, 
                                                                                  finalVocabularyAsDict)
        if discardTheArticle == True:
            return None, None, None, True
        wordsListWithTAGSList.append(wordsListWithTAGS)
        wordLevelEmbeddingsAsTensorList.append(wordLevelEmbeddingsAsTensor)
    
    return wordsListWithTAGSList, wordLevelEmbeddingsAsTensorList, unknownWordsDictionaryForAnArticle, False # the article was not discarded
        
    

    
'''
INPUT:
- highlights/target sentences as single string
- unknownWordsDictionaryForAnArticle built when processing source article

the function preprocess the target sentences (higlights or identical input sequences if AutoEncoder transformer)
such that each word is represented as one-hot encoding according to finalVocabularyAsDict
If OOV word we see if there exist the corresponding UNKTag from the unknownWordsDictionaryForAnArticle,
which was built when processing the input article
If a word does not belong to the final vocabulary nor to the unknownWordsDictionaryForAnArticle we use [UNK4BOTHDICT] tag
We replace '.' with [SEP] tag if the sequence is followed by another sequence,
we replace '.' with [EOS] tag if it is the last sequence (i.e. summary end)

RETURNS:

'''
def preprocessHighlights(highlights, unknownWordsDictionaryForAnArticle, finalVocabularyAsDict):
    highlights = expandContractions(highlights)
    highlights = splitSentences(highlights) #  \n is deleted when splitting into sentences
    
    # replacing '.' with [SEP] or [EOS]
    sentencesWithFinalToken = []
    for i, sentence in enumerate(highlights):
        if i != len(highlights)-1:
            new_sentence = sentence[:-1] + ' ' + 'SEP' # if there is another sentence following
        else:
            new_sentence = sentence[:-1] + ' ' + 'EOS' # if the summary has ended
        sentencesWithFinalToken.append(new_sentence)
            
    # NB: tokenization of target sentences is made at word level i.e. the same sentence will have different length using input or output encoding
    
    encodedSequences = [] # list of tensors where each tensor (#words, #len(finalVocabularyAsDict)-1)
    for sentence in sentencesWithFinalToken:
        wordsInSentence = word_tokenize(sentence)
        listOfIds = []
        for word in wordsInSentence:
          # correcting some artifacts that make '' appear `` sometimes
          if word == "``":
            word = "''"
          if word == 'SEP' or word == 'EOS':
            index = list(finalVocabularyAsDict).index('['+word+']')
            listOfIds.append(index)
          # is LONDON to lower belongs to final vocabulary we use it
          elif word.lower() in finalVocabularyAsDict:
            index = list(finalVocabularyAsDict).index(word.lower())
            listOfIds.append(index)
          else:
            if word in unknownWordsDictionaryForAnArticle:
              unknownTag = unknownWordsDictionaryForAnArticle[word]
              # we now retrive position in finalVocabularyAsDict x
              index = list(finalVocabularyAsDict).index(unknownTag)
              listOfIds.append(index)
            else:
              index = list(finalVocabularyAsDict).index('[UNK4BOTHDICT]')
              listOfIds.append(index)
        
        wordsIds = torch.tensor(listOfIds)
        oneHotMatrix = F.one_hot(wordsIds, num_classes=len(finalVocabularyAsDict))
        encodedSequences.append(oneHotMatrix)
        
        encodedSequencesAsListOfSingleEncoding = [] #  target sentences starting with START TAG
        # appending [START] token embedding
        startEmbedding = F.one_hot(torch.tensor([list(finalVocabularyAsDict).index('[START]')]), num_classes=len(finalVocabularyAsDict))
        encodedSequencesAsListOfSingleEncoding.append(startEmbedding[0])
        
        for tensor in encodedSequences:
          for encoding in tensor:
            encodedSequencesAsListOfSingleEncoding.append(encoding)

    return torch.stack(encodedSequencesAsListOfSingleEncoding).type(torch.float16) # tensor (#words all sequences, #dictionary length)
        

''' Retrieve original sentence(s) from output encoded version '''
def fromFinalVocabularyEncodingBackToWords(oneHotMatrix, finalVocabularyAsDict, unknownWordsDictionaryForAnArticle):
    convertedSentences = ''
    inverse_unknownWordsDictionaryForAnArticle = {v: k for k, v in unknownWordsDictionaryForAnArticle.items()}
    for item in oneHotMatrix:
        key = torch.argmax(item).item() # position in the dictionary
        word = list(finalVocabularyAsDict)[key] # key value at that position
        # if it is UNK and not UNK4BOTHDICT we look into unknownWordsDictionaryForAnArticle
        if word[0:4] == '[UNK' and word != '[UNK4BOTHDICT]':
            correspondingWord = inverse_unknownWordsDictionaryForAnArticle[word]
        elif word == '[UNK4BOTHDICT]':
            correspondingWord = 'UNK'
        else:
            correspondingWord = word
        convertedSentences = convertedSentences + correspondingWord + ' '
        
        # we need to replace [SEP] and [EOS] with '.'
        # convert to upper case words after a dot
        
    return convertedSentences # as string






''' MAIN '''
# Load dataset
# load tokenizer
# load bert model
# load Dictionary
# load dictionary of embedding for each UNK tag

# nltk.download('punkt')

# For a article-highlights pair:

# preprocessArticle
# use dictionary of OOV words to encode words in target sentences using same UNKtag (wrt to each vocabulary)

