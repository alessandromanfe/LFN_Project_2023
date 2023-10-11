''' Building Preprocessed Dataset '''

import torch
import numpy as np
import pickle5 as pickle
import math
import os
import nltk

import sys
sys.path.insert(0, './src/')
import libraryForBuildingDatasetOptimized as l4bdOptimized # add also all dependencies required by this defined by us library!

def preprocessArticleHighlightsPair(cnn_dailymail_dataset, TrainValidationOrTest, articleNumber, tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS):
    article = cnn_dailymail_dataset[TrainValidationOrTest][articleNumber]['article']
    highlights = cnn_dailymail_dataset[TrainValidationOrTest][articleNumber]['highlights']
    # preprocess source article using input tokenization and embedding
    wordsListWithTAGSList, wordLevelEmbeddingsAsTensorList, unknownWordsDictionaryForAnArticle, discardTheArticle = l4bdOptimized.preprocessArticle(article, tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS)
    sourceArticleInputEmbedding = {'wordsListWithTAGSList' : wordsListWithTAGSList, 'wordLevelEmbeddingsAsTensorList' : wordLevelEmbeddingsAsTensorList, 'unknownWordsDictionaryForAnArticle' : unknownWordsDictionaryForAnArticle}
    if discardTheArticle == True:
        return None, None, None, True
    return sourceArticleInputEmbedding, article, highlights, discardTheArticle


def constructPreprocessedDataset(cnn_dailymail_dataset, trainValidationOrTest, numberOfInstances, startingFromArticleIndex, pathToSavePickle, tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS):
    # check if folders exists otherwise create them, we rename all after
    for i in range(startingFromArticleIndex, startingFromArticleIndex + numberOfInstances):
        fileName = 'moreValidation' + '/' + str(i) + '_BeforeRenaming.pickle'
        try:
            sourceArticleInputEmbedding, article, highlights, discardTheArticle = preprocessArticleHighlightsPair(cnn_dailymail_dataset, trainValidationOrTest, i, tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS)
            if discardTheArticle != True:
                with open(pathToSavePickle + fileName, 'wb') as handle:
                    pickle.dump( (sourceArticleInputEmbedding, article, highlights) , handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pass
        if i % 100 == 0:
            print(f"Preprocessed up to article {i} of {trainValidationOrTest}.")


def main():

    nltk.download('punkt', download_dir='/nfsd/VFdisk/zamaiandre/DocSumProject/punkt')
    nltk.data.path.append('/nfsd/VFdisk/zamaiandre/DocSumProject/punkt')

    FINAL_VOCABULARY_SIZE = 9998
    MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS = 100
    PATH_TO_TOP_FREQWORDS = './data/topEnglishWords.csv'

    cnn_dailymail_dataset = l4bdOptimized.loadCnnDaylyMailDataset()
    tokenizer = l4bdOptimized.loadPretrainedBERTmodelTokenizer()
    model = l4bdOptimized.loadPretrainedBERTmodel()
    finalVocabularyAsDict = l4bdOptimized.loadTargetVocabulary(FINAL_VOCABULARY_SIZE, MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS, PATH_TO_TOP_FREQWORDS)
    dictionaryOfEmbeddingsForUnknownTAGS = l4bdOptimized.defineDictionaryOfEmbeddingsForUnknownTAGS(MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS)    
    print("Starting dataset construction")
    #constructPreprocessedDataset(cnn_dailymail_dataset, 'train', 20000, 10000, './data/PreprocessedDataset/', tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS)
    #constructPreprocessedDataset(cnn_dailymail_dataset, 'validation', 500, 0, './data/PreprocessedDataset/', tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS)
    #constructPreprocessedDataset(cnn_dailymail_dataset, 'test', 500, 0, './data/PreprocessedDataset/', tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS)
    constructPreprocessedDataset(cnn_dailymail_dataset, 'validation', 5000, 501, './data/PreprocessedDataset/', tokenizer, model, finalVocabularyAsDict, dictionaryOfEmbeddingsForUnknownTAGS)

if __name__ == "__main__":
    main()
    print("Preprocessed Dataset created")
