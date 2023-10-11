import torch
#import torch.nn.functional as F
#from transformers import BertTokenizer, BertModel
#import numpy as np
#import pandas as pd
from torch.utils.data import DataLoader, Dataset
#from torch_geometric.utils import degree

#from datasets import load_dataset
import contractions
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

#import nltk
import torch
#import numpy as np
from sentence_transformers import SentenceTransformer, util
#import random
import pickle
#import networkx as nx
#import matplotlib.pyplot as plt
#from networkx.drawing.nx_pydot import graphviz_layout
#import math
import libraryForBuildingDatasetOptimized as l4bdOptimized
from more_x2_Complex_encDec_multiGPU import createModel
from more_x2_Complex_encDec_multiGPU import unpack_inputSentencesAsTensorBatch_whichArePaddingWordsMaskBatch

import sys
import os
import traceback

PATH_TO_DATASET= '/nfsd/VFdisk/zamaiandre/DocSumProject/data/PreprocessedDataset/'
PATH_OUTPUT = '/nfsd/VFdisk/zamaiandre/DocSumProject/data/GraphsDataset/'
PATH_TO_CSV_OF_TOPENGLISHFREQWORDS = '/nfsd/VFdisk/zamaiandre/DocSumProject/data/topEnglishWords.csv' # csv with top frequent English words
PATH_TO_CHECKPOINT = './src/transfPretrainCP_multiGPU_more_x2_Complex.tar'

# these 2 variables MUST be same of the ones used to preprocess the articles to produce PreprocessedDataset
FINAL_VOCABULARY_SIZE = 9998 # special characters, UNK tags and tokens like [START] [EOS] exluded --> will be ca 10136 (NUMBER_EXPECTED_FEATURES_OUTPUT)
MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS = 100 # make sure number of UNK tags is equal to the one with which the articles were preprocessed 

MAX_NUMBER_OF_WORDS_IN_SENTENCE = 50 # a sentence in input will be padded or trunked to this fixed length (required by transformer architecture)
MAX_NUMBER_OF_SENTENCES_AE = 5 # number of sentences fed separately in input to the decoder and required in output by the decoder

NUMBER_EXPECTED_FEATURES_INPUT = 770 # 768 from BERT embedding + 2 added by us for UNK tags
NUMBER_EXPECTED_FEATURES_OUTPUT = FINAL_VOCABULARY_SIZE + 138 # (+138 takes into account UNKi tags, special tokens and special characters added when building FinalVocabulary) = 10136

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


''' Build graph from article sentences '''
def generateGraph(sentencesModel, sentences, device, maxEdge=0, minSplit=4):
    indices = [i for i,s in enumerate(sentences) if len(s.split())>minSplit]
    if len(indices) == 0:
        return None
    validSentences = [sentences[i] for i in indices]
    sentEmb = [sentencesModel.encode(s, convert_to_tensor=True) for s in validSentences]
    sentEmb = torch.stack(sentEmb)
    cosSimMatrix = util.pytorch_cos_sim(sentEmb,sentEmb)
    triuCSM = torch.tril(cosSimMatrix, diagonal=-1)
    meanCS = triuCSM[triuCSM>0].mean()
    triuCSM[triuCSM<meanCS] = 0
    edgeIndex = triuCSM.nonzero()
    edgeWeight = triuCSM[triuCSM.nonzero(as_tuple=True)].flatten()
    edgeWeight, sortIndices = torch.sort(edgeWeight, descending=True, stable=True)
    edgeIndex = edgeIndex[sortIndices].transpose(0,1)

    if maxEdge == 0:
        #Default value (better behaviour)
        maxEdge = 3*len(validSentences)

    numEdge = min(maxEdge,len(edgeWeight))

    validNodes = torch.unique(edgeIndex.flatten(), sorted=True)
    nodesMask = list(range(len(indices)))
    invalidNodes = [n for n in nodesMask if n not in validNodes]
    for nn in invalidNodes:
        for i in range(nn,len(nodesMask)):
            nodesMask[i] -= 1
    edgeIndex = torch.tensor(nodesMask).to(device)[edgeIndex]
    validIndices = [indices[i] for i in validNodes]
    validConnectedSentences = [validSentences[i] for i in validNodes]


    return edgeIndex[:,:numEdge], edgeWeight[:numEdge], validIndices, validConnectedSentences



class PickleDatasetReader(Dataset):
    def __init__(self, pathToFolder, trainValidationOrTest):
        self.pathToFolder = os.path.join(pathToFolder, trainValidationOrTest) # eg './data/PreprocessedDataset' , 'train'
        self.finalVocabularyAsDict = l4bdOptimized.loadTargetVocabulary(FINAL_VOCABULARY_SIZE, MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS, path=PATH_TO_CSV_OF_TOPENGLISHFREQWORDS) 
        self.maxNumberOfWordsInSentence = MAX_NUMBER_OF_WORDS_IN_SENTENCE
        self.maxNumberOfSentences = MAX_NUMBER_OF_SENTENCES_AE
        self.trainValidationOrTest = trainValidationOrTest
        nltk.data.path.append('/nfsd/VFdisk/zamaiandre/DocSumProject/punkt')
        nltk.data.path.append('./punkt/tokenizers/punkt')
    def __len__(self):
        if '.DS_Store' in os.listdir(self.pathToFolder):
            return len(os.listdir(self.pathToFolder)) -1
        else:
            return len(os.listdir(self.pathToFolder))

    def __getitem__(self, idx):
        # load the preprocessed article from pickle file (for each idx there is a file with source article preprocessed)
        preprocessedDatasetReloaded = None
        pathToArticle = os.path.join(self.pathToFolder, str(idx) + '.pickle')
        with open(pathToArticle, 'rb') as handle:
            preprocessedDatasetReloaded = pickle.load(handle)

        # it is a tuple containing (data, originalArticle as string, highlights)
        data, originalArticle, highlightsOriginal = preprocessedDatasetReloaded

        expandedContractionsArticle = l4bdOptimized.expandContractions(originalArticle)
        sentencesInOriginalArticleList = l4bdOptimized.splitSentences(expandedContractionsArticle)

        # unwrap data dictionary  
        wordsListWithTAGSList = data['wordsListWithTAGSList']
        wordLevelEmbeddingsAsTensorList = data['wordLevelEmbeddingsAsTensorList']
        unknownWordsDictionaryForAnArticle = data['unknownWordsDictionaryForAnArticle']

        wordEmbeddingBatchToEncode = []
        wordPaddingBatchToEncode = []

        embSize = len(wordLevelEmbeddingsAsTensorList)
        batch = embSize//self.maxNumberOfSentences + int(embSize%self.maxNumberOfSentences > 0)
        for k in range(batch):
            numberOfSentences = min(self.maxNumberOfSentences,(embSize - self.maxNumberOfSentences*k))
            sentencesToPick = range(self.maxNumberOfSentences*k,self.maxNumberOfSentences*k+numberOfSentences)

            # we have a list of tensors where each one is the embedding of the picked sentences, we don't stack them as will be fed separately to encoder
            inputSentencesTensorsList = [wordLevelEmbeddingsAsTensorList[i].type(torch.float32) if i < len(wordLevelEmbeddingsAsTensorList) else torch.zeros((MAX_NUMBER_OF_WORDS_IN_SENTENCE, NUMBER_EXPECTED_FEATURES_INPUT)).type(torch.float32) for i in sentencesToPick] # converting from float16 back to float32
            
            
            # to enable batching and speed up computation we cannot keep inputSentencesTensorsList even if we would like to keep input sentences separated to be fed individually to the encoder
            # moreover we need to have them all to same dimension

            # we pad all tensors to have same shape[0] st each tensor MAX_NUMBER_OF_WORDS_IN_SENTENCE=50 x 770 or cut if dim0 greater than max allowed
            # we need to have like a mask [1, 1, 1, 1, 0] so we can understand which are padding elements and which are not
            # during forward pass we can split them again splitting/grouping by MAX_NUMBER_OF_WORDS_IN_SENTENCE

            listOfPaddedTensors = []
            whichArePaddingWordsMaskList = []
            for tensor in inputSentencesTensorsList:
                # if dim0 greater than self.maxNumberOfWordsInSentence we truncate
                if tensor.shape[0] > self.maxNumberOfWordsInSentence:
                    tensorPadded = tensor[:self.maxNumberOfWordsInSentence, :]
                    listOfPaddedTensors.append(tensorPadded)
                    whichArePaddingWordsMaskList.append(torch.ones(tensorPadded.shape[0])) # all ones
                else:
                    paddingBottom = self.maxNumberOfWordsInSentence - tensor.shape[0]
                    padding = torch.nn.ZeroPad2d((0,0,0,paddingBottom))
                    tensorPadded = padding(tensor)
                    listOfPaddedTensors.append(tensorPadded)
                    whichArePaddingWordsMaskList.append(torch.cat((torch.ones(tensor.shape[0]), torch.zeros(paddingBottom)))) # 0 if it is padding

            inputSentencesAsTensor = torch.cat(listOfPaddedTensors)
            # we now pad with all ZEROs st all input instances has same length
            # since we decided to have in input at most 5 sentences each tensor will be padded to be 5*MaxNumWords x 770
            inputSentencesAsTensor = torch.cat((inputSentencesAsTensor, torch.zeros((self.maxNumberOfSentences * self.maxNumberOfWordsInSentence - inputSentencesAsTensor.shape[0], NUMBER_EXPECTED_FEATURES_INPUT))))
            # same for whichArePaddingWordsMask
            whichArePaddingWordsMask = torch.cat(whichArePaddingWordsMaskList)
            whichArePaddingWordsMask = torch.cat((whichArePaddingWordsMask, torch.zeros(self.maxNumberOfSentences * self.maxNumberOfWordsInSentence - whichArePaddingWordsMask.shape[0])))

            wordEmbeddingBatchToEncode.append(inputSentencesAsTensor.type(torch.float32))
            wordPaddingBatchToEncode.append(whichArePaddingWordsMask.type(torch.int))
                
        return (wordEmbeddingBatchToEncode, 
                wordPaddingBatchToEncode, 
                embSize, 
                sentencesInOriginalArticleList,
                wordsListWithTAGSList, 
                unknownWordsDictionaryForAnArticle, 
                highlightsOriginal, 
                originalArticle, 
                str(pathToArticle))


def main(train_test_validation, num_sample=0):

    TRAIN_TEST_VALID = train_test_validation #'validation'
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(str(device))
    checkpoint = torch.load(PATH_TO_CHECKPOINT, map_location= (device))
    model = createModel(device)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    testData = PickleDatasetReader(PATH_TO_DATASET,TRAIN_TEST_VALID)
    testDataLoader = DataLoader(testData, batch_size=1, shuffle=False)

    sentencesModel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vocabulary = testData.finalVocabularyAsDict

    finalDataset = []
    count = 0
    skipped = 0

    for (wordEmbeddingBatchToEncode, 
        wordPaddingBatchToEncode, 
        embSize, 
        sentencesInOriginalArticleList, 
        wordsListWithTAGSList,
        unknownWordsDictionaryForAnArticle, 
        highlightsOriginal, 
        originalArticle, 
        path) in testDataLoader:

        WEBTESplitted = []
        WPBTESplitted = []
        for emb, pad in zip(wordEmbeddingBatchToEncode, wordPaddingBatchToEncode):
            embSplitted, padSplitted = unpack_inputSentencesAsTensorBatch_whichArePaddingWordsMaskBatch(emb, pad)
            WEBTESplitted.append(embSplitted)
            WPBTESplitted.append(padSplitted)
        WEBTESplitted = torch.cat(WEBTESplitted).to(device)
        WPBTESplitted = torch.cat(WPBTESplitted).to(device)
        with torch.no_grad():
            encodedSentences = model.encodeSentences(WEBTESplitted,WPBTESplitted)
        encodedSentences = encodedSentences.flatten(0,1)[:embSize,:]

        if len(unknownWordsDictionaryForAnArticle)>0 and isinstance(list(unknownWordsDictionaryForAnArticle.values())[0],(list,tuple)):
            invertedDictionary = {v[0]:k for k,v in unknownWordsDictionaryForAnArticle.items()}
        else:
            invertedDictionary = {v:k for k,v in unknownWordsDictionaryForAnArticle.items()}

        parsedSentences = []
        for s in wordsListWithTAGSList:
            newsent = ""
            for w in s:
                if isinstance(w,(list,tuple)):
                    w = w[0]
                if len(w) > 5 and w[:4] == "[UNK":
                    if w in invertedDictionary:
                        newsent += invertedDictionary[w] + " "
                elif len(w) > 4 and w == "[SEP]":
                    newsent += "."
                else:
                    newsent += w + " "
            parsedSentences.append(newsent)

        if len(parsedSentences) != encodedSentences.size()[0]:
            print(f"Warning: {path}\nDiscrepancy in size BEFORE removing dots: "
                  f"sentences: {len(sentencesInOriginalArticleList)}, "
                  f"encodedSentences: {encodedSentences.size()[0]}.",flush=True)
        
            parsedSentences = [s for s in parsedSentences if s[0].count('.') != len(s[0])]

        if len(parsedSentences) != encodedSentences.size()[0]:
            print(f"Warning: {path}\nDiscrepancy in size AFTER removing dots: "
                  f"sentences: {len(sentencesInOriginalArticleList)}, "
                  f"encodedSentences: {encodedSentences.size()[0]}.",flush=True)
            print("Skipped.",flush=True)
            skipped+=1
            continue

        try:
            genResult = generateGraph(sentencesModel, parsedSentences, device)
            if genResult is None:
                print(f"Warning: {path}\nInsufficent valid sentences for the article",flush=True)
                print(f"Skipped.",flush=True)
                skipped+=1
                continue
            edgeIndex, edgeWeight, indices, sentences = genResult

        except Exception as e:
            print(f"Warning: {path}\nError during graph generation (probably too small article)",flush=True)
            print("Skipped.",flush=True)
            print(e)
            print(traceback.format_exc())
            print(f"Parsed sentences: {parsedSentences}")
            skipped+=1
            break
            #continue
            #print(path)
            #print(encodedSentences.size())
            #print(traceback.format_exc())
            #print(count)
            #break
        
        if len(indices)==0 or encodedSentences.size()[0]==0:
            print(f"Warning: {path}\nIndices or encoding list empty\nSkipped",flush=True)
            skipped+=1
            continue
        if indices[-1] >=  encodedSentences.size()[0]:
            print(f"Warning: {path}\nDiscrepancy in size AFTER building graph: "
                  f"biggest index: {indices[-1]}, "
                  f"encodedSentences: {encodedSentences.size()[0]}.",flush=True)
            print("Skipped.",flush=True)
            skipped+=1
            continue
        
        try:
            validEncodedSentences = encodedSentences[indices]
        except Exception as e:
            print(path)
            print(sentences[0])
            print(indices)
            print(encodedSentences.size())
            print(traceback.format_exc())
            print(count)
            break

        data = {"graphData":{"nodeEmbedding": validEncodedSentences,
                                  "edgeIndex":edgeIndex,
                                  "edgeWeight":edgeWeight},
                     "sentences":sentences,
                     "originalArticle":originalArticle,
                     "highlights":highlightsOriginal,
                     "UnknownVocabulary":unknownWordsDictionaryForAnArticle}

        filePath = os.path.join(PATH_OUTPUT, TRAIN_TEST_VALID ,str(count) + '.pickle')

        with open(filePath,'wb') as handle:
            pickle.dump( data , handle, protocol=pickle.HIGHEST_PROTOCOL)

        #finalDataset.append(data)

        count+=1
        if num_sample != 0 and count>=num_sample:
            print(count)
            break

        if count%100 == 0:
            print(f"Number of articles currently converted: {count},   skipped: {skipped}.", flush=True)
    '''
    fileName = "finalDataset.pickle"
    with open(PATH_OUTPUT + fileName, 'wb') as handle:
        pickle.dump( finalDataset , handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    '''
    with open(PATH_OUTPUT + fileName, 'rb') as handle:
        ds = pickle.load(handle)
        ds0 = ds[0]
    '''
    print(f"Done.\nSuccesfully converted {count} articles ({skipped} skipped).")

        



if __name__ == "__main__":
    n = len(sys.argv)
    if n < 2:
        print("Not enough argument: [train_test_validation] [num_sample]")
    if n == 2:
        ttv = sys.argv[1]
        main(ttv)
    if n == 3:
        ttv = sys.argv[1]
        num = int(sys.argv[2])
        main(ttv,num)
