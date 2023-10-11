'''

MORE COMPLEX MODEL

- Between Extractive & Abstractive Document Summarization Project 
- Andrew Zamai & Alessandro Manfè
- LFN course 2022/23, Prof. Fabio Vandin

Pretraining Encoder & Decoder transformers weights before deploying them in the final GNN model (as described in the project report).

# Workspace preparation:
- place this file inside DocSumProejct/src and make sure job HOME path is at DocSumProejct level
-Install required libraries below and any other libraries used in libraryForBuildingDatasetOptimized.py
-Install torch with CUDA version if GPU available.
-Load libraryForBuildingDatasetOptimized.py into workspace and add path to sys;
-Load csv of topEnglishFrequentWords and change path to it;
-check to have PreprocessedDataset with train/validation/test folders and numerated files

'''


''' --------------------------------------------- IMPORTING REQUIRED LIBRARIES --------------------------------------------- '''
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import pickle
import random
import math
import os
import nltk

# multi-GPUs
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import sys
sys.path.insert(0, './src/') # change path to folder where library is placed
import libraryForBuildingDatasetOptimized as l4bdOptimized


''' ------------------------------------------ DistributedDataParallelSetup ------------------------------------------ '''
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process # assigned automatically 
        world_size: Total number of processes # 1 process on each GPU --> will be number of GPUs available
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


''' ------------------------------------------ HERE WE DEFINE SOME GLOBAL VARIABLES ------------------------------------------- '''

BATCH_SIZE = 64  # adjust according to GPU available 

PATH_TO_CSV_OF_TOPENGLISHFREQWORDS = './data/topEnglishWords.csv' # csv with top frequent English words
PATH_TO_DATASET = './data/PreprocessedDataset'

# these 2 variables MUST be same of the ones used to preprocess the articles to produce PreprocessedDataset
FINAL_VOCABULARY_SIZE = 9998 # special characters, UNK tags and tokens like [START] [EOS] exluded --> will be ca 10136 (NUMBER_EXPECTED_FEATURES_OUTPUT)
MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS = 100 # make sure number of UNK tags is equal to the one with which the articles were preprocessed 

MAX_NUMBER_OF_WORDS_IN_SENTENCE = 50 # a sentence in input will be padded or trunked to this fixed length (required by transformer architecture)
MAX_NUMBER_OF_SENTENCES_AE = 5 # number of sentences fed separately in input to the decoder and required in output by the decoder

NUMBER_EXPECTED_FEATURES_INPUT = 770 # 768 from BERT embedding + 2 added by us for UNK tags
NUMBER_EXPECTED_FEATURES_OUTPUT = FINAL_VOCABULARY_SIZE + 138 # (+138 takes into account UNKi tags, special tokens and special characters added when building FinalVocabulary) = 10136



''' ----------------------------------------------- DATASET PREPARATION ----------------------------------------------- '''

class PreprocessedDatasetForAE(Dataset):
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
        # unwrap data dictionary  
        wordsListWithTAGSList = data['wordsListWithTAGSList']
        wordLevelEmbeddingsAsTensorList = data['wordLevelEmbeddingsAsTensorList']
        unknownWordsDictionaryForAnArticle = data['unknownWordsDictionaryForAnArticle']

        if self.trainValidationOrTest == 'train':
          # with p=0.5 we use first 5 sentences
          # with p=0.5 we pick some random sentences
          p = np.random.uniform(0.0, 1.0)
          if p <= 0.5:
            numberOfSentences = 5
            sentencesToPick = [0, 1, 2, 3, 4]
          else:
            numberOfSentences = 5
            sentencesToPick = [random.randint(0, len(wordLevelEmbeddingsAsTensorList)-1) for i in range(numberOfSentences)] # eg [0, 23, 52]
            sentencesToPick.sort() # ordering sentences (important for UNK tags order)
        # if validation or test we look always at same sentences to have an objective and not changing metric
        else:
          numberOfSentences = 5
          sentencesToPick = [0, 1, 2, 3, 4]

        # we have a list of tensors where each one is the embedding of the picked sentences, we don't stack them as will be fed separately to encoder
        inputSentencesTensorsList = [wordLevelEmbeddingsAsTensorList[i].type(torch.float32) if i < len(wordLevelEmbeddingsAsTensorList) else torch.zeros((MAX_NUMBER_OF_WORDS_IN_SENTENCE, NUMBER_EXPECTED_FEATURES_INPUT)).type(torch.float32) for i in sentencesToPick] # converting from float16 back to float32
        
        # we now want to pick the same sentences from the article, concatenate them and preprocess them as target
        expandedContractionsArticle = l4bdOptimized.expandContractions(originalArticle)
        sentencesInOriginalArticleList = l4bdOptimized.splitSentences(expandedContractionsArticle)
        originalSentencesPickedToBeTarget = [sentencesInOriginalArticleList[i] if i < len(sentencesInOriginalArticleList) else  ' ' for i in sentencesToPick]
        targetSentence = ''
        for sent in originalSentencesPickedToBeTarget:
            targetSentence += sent
            # place space when concatenating them
            if sent[-1] == '.':
                targetSentence += ' '

        # if corrupted article with 0 sentences minimum sentence is [START] [EOS]
        # START added in preprocessHighlights function, EOS at the end so we just set targetSentence = ' '
        if len(targetSentence) <= 5:
          targetSentence = ' '

        # unknownWordsDictionaryForAnArticle has to be the same produced when preprocessing source article
        targetSentencesEmbedded = l4bdOptimized.preprocessHighlights(targetSentence, unknownWordsDictionaryForAnArticle, self.finalVocabularyAsDict)

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

        # as we padded for input we need to pad also target sentences
        # the target is obtained by concatenating all sentences and embedding them in a single matrix of dimension (#words, finalVocabularyLen)
        # we need to pad along dim=0, we may chose as dim0 = #self.maxNumberOfWordsInSentence * self.maxNumberOfSentences
        if targetSentencesEmbedded.shape[0] > self.maxNumberOfSentences * self.maxNumberOfWordsInSentence:
            # if too long we retain maxNRows -1 and add last the embedding of EOS token
            targetSentencesEmbeddedPadded = targetSentencesEmbedded[0: (self.maxNumberOfSentences * self.maxNumberOfWordsInSentence) -1, :]
            targetSentencesEmbeddedPadded = torch.cat((targetSentencesEmbeddedPadded, torch.unsqueeze(targetSentencesEmbedded[-1, :], dim=0))) # retrieving EOS embedding from last position of targetSentencesEmbedded
            whichArePaddingWordsTargetSentence = torch.ones(targetSentencesEmbeddedPadded.shape[0]) # all ones since all valid
        else:
            paddingBottom = self.maxNumberOfWordsInSentence * self.maxNumberOfSentences - targetSentencesEmbedded.shape[0]
            padding = torch.nn.ZeroPad2d((0,0,0,paddingBottom))
            targetSentencesEmbeddedPadded = padding(targetSentencesEmbedded)
            whichArePaddingWordsTargetSentence = torch.cat((torch.ones(targetSentencesEmbedded.shape[0]), torch.zeros(paddingBottom)))
            
        return inputSentencesAsTensor.type(torch.float32), whichArePaddingWordsMask.type(torch.int), targetSentencesEmbeddedPadded.type(torch.float32), whichArePaddingWordsTargetSentence.type(torch.int)


''' ----------------------------------------------- HELPER CLASSES & MODEL DEFINITION ----------------------------------------------- '''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor, shape [batch_size, seq_len , embedding_dim]
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz, DEVICE):
    # take upper triangular part of matrix of all 1 (output still a matrix but lower triangular part are zeros)
    # transpose st 0 are now on the upper triangular part
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    # replace 0 in upper triangular part with -infinity and initialize lower triangular part to 0
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_masks_for_input(inputEmbeddingsForSingleSentenceBatched, whichArePaddingMaskBatched, DEVICE):
    batch_size = inputEmbeddingsForSingleSentenceBatched.shape[0]
    src_seq_len = inputEmbeddingsForSingleSentenceBatched.shape[1] 
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (whichArePaddingMaskBatched == 0).to(DEVICE) 
    return src_mask, src_padding_mask


def create_masks_for_output(embeddingForTargetSentenceBatched, whichArePaddingMaskBatched, DEVICE):
    batch_size = embeddingForTargetSentenceBatched.shape[0]
    tgt_seq_len = embeddingForTargetSentenceBatched.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    tgt_padding_mask = (whichArePaddingMaskBatched == 0).to(DEVICE)
    return tgt_mask, tgt_padding_mask


def create_memory_key_pad_mask(whichArePaddingWordsMaskSplittedBatch):
    batch_size = whichArePaddingWordsMaskSplittedBatch.shape[0]
    # we sum over last dimension 
    # if count !=0 we mark it as valid sentence
    # in practice during training always all valid (all False)
    memory_key_pad_mask = torch.sum(whichArePaddingWordsMaskSplittedBatch, dim=-1)
    return (memory_key_pad_mask == 0)


''' MODEL DEFINITION:
Remember: input sentences are passed individually 1 by 1, their condensed representations stacked at the bottleneck and fed all together to the decoder.
NB: since the encoder works stackin N encoder layers the output dim of the encoder will be (source_emb_size x 50 x 770)(has to be the same as the input to be fed potentially into another encoder layer)
From matrix representing a sentence (50 x 770) to a single vector embedding but of same "width" (770) we need to perform some manipulation (eg mean to collapse 1 dimension)
'''
class AETransformer(nn.Module):

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 source_emb_size: int,
                 nheadEncoder: int,
                 target_emb_size : int,
                 target_emb_size_reduced : int,
                 nheadDecoder: int,
                 dropout: float = 0.2, 
                 device : int = 0):
        super(AETransformer, self).__init__()
        self.DEVICE = device

        # the encoder stack of layers works on a single sentence representation at the time (BATCH_SIZE x MAX_NUM_WORDS=50 x source_emb_size=770)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=source_emb_size, 
                                                                        nhead=nheadEncoder,
                                                                        dim_feedforward=512,
                                                                        dropout=0.2,
                                                                        activation='relu',
                                                                        layer_norm_eps = 1e-5,
                                                                        batch_first = True,
                                                                        norm_first = False),
                                            num_layers=num_encoder_layers)
                                            #enable_nested_tensor = False)
        
        # NB: since the decoder takes in input both the encoder hidden representations and the target (autoregressive)
        # we have to reduce the one-hot encoding to smaller vectors, will be = to target_emb_size_reduced = source_emb_size
        # where target_emb_size_reduced is the reduced learned target representation (from onehot of len(finalVoc) to smaller encoding)
        self.reduceTargetEmbDimNN = nn.Sequential(nn.Linear(target_emb_size, target_emb_size_reduced), nn.ReLU())

        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=target_emb_size_reduced,
                                                                        nhead=nheadDecoder,
                                                                        dim_feedforward=512,
                                                                        dropout=0.2,
                                                                        activation='relu',
                                                                        layer_norm_eps = 1e-5,
                                                                        batch_first = True,
                                                                        norm_first = False),
                                            num_layers=num_decoder_layers)
        
        # positional_encoding_sourceSentence works on a single sentence representation at the time of length MAX_NUMBER_OF_WORDS_IN_SENTENCE
        self.positional_encoding_sourceSentence = PositionalEncoding(source_emb_size, 0.2, MAX_NUMBER_OF_WORDS_IN_SENTENCE)
        # positional_encoding_targetSentence works on entire target sentence (obtained concatenating the sentences in input)
        self.positional_encoding_targetSentence = PositionalEncoding(target_emb_size_reduced, 0.2, MAX_NUMBER_OF_WORDS_IN_SENTENCE * MAX_NUMBER_OF_SENTENCES_AE)

        # from target_emb_size_reduced (decoder output dim) back to one hot vector len(finalVoc)
        self.generator = nn.Linear(target_emb_size_reduced, target_emb_size) # NB no need for softmax or other non linear functions as the crossentropy loss in pytorch already includes computing softmax


    # inputSentencesAsTensorsSplittedBatch has dimension (BATCH_DIM, MAX_N_SEQ=5, MAX_N_WORDS=50, INPUTDIM=770)
    # during forward we pass to the encoder (BATCH_DIM, MAX_N_WORDS=50, INPUTDIM=770) for each of the 5 sequences
    # the whichArePaddingWordsMaskSplittedBatch will allow to create src_key_padding mask to not let attend to padding tokens
    def forward(self,
                inputSentencesAsTensorsSplittedBatch: Tensor,
                whichArePaddingWordsMaskSplittedBatch: Tensor,
                targetSentencesEmbeddedPaddedBatch : Tensor, 
                whichArePaddingWordsTargetSentenceBatch : Tensor,
                memory_key_pad_mask: Tensor):
        
        # list of 1 tensor for each sentence (BATCHDIM, 50, 770)
        sentencesList = [ inputSentencesAsTensorsSplittedBatch[:, i, :, :] for i in range(MAX_NUMBER_OF_SENTENCES_AE) ]
        inputMasksList = [ whichArePaddingWordsMaskSplittedBatch[:, i, :] for i in range(MAX_NUMBER_OF_SENTENCES_AE) ]

        # here we append hidden representation from the encoder to stack them vertically
        encodedSentenceSUM = []
        # in order in the final model to allow to use a varible node-representations from GNN output (eg top k degree nodes) (but always <=5) 
        # we always stack vertically 5 node-representations but mask the "not-true ones"
        for sentence, mask in zip(sentencesList, inputMasksList):
            # create src_mask, key_padding mask
            src_mask, src_padding_mask = create_masks_for_input(sentence, mask, self.DEVICE)
            # add positional encoding
            sentence = self.positional_encoding_sourceSentence(sentence)  
            # pass throw encoder
            encodedSentence = self.encoder(sentence, src_mask, src_padding_mask)
            # now encodedSentence has shape (BATCH_SIZE, MAX_NUM_WORDS, 770) (has has to be input for any other encoder layer when stacking)
            # we want the output of the encoder module to have dimension (BATCH_SIZE, 770) 
            # each sentence represented through a matrix is now repr. through a vector
            # what we can collapse matrix in 1 vector by taking mean along dim=0
            encodedSentence_collapsed = torch.mean(encodedSentence, 1) # collapsing dim 1
            encodedSentenceSUM.append(encodedSentence_collapsed) # stacking representations vertically (BATCH_SIZE, NUMBER OF SENTENCES, 770)
            
        encodedSentenceSUM = (torch.stack(encodedSentenceSUM)).to(self.DEVICE, torch.float32)
        encodedSentenceSUM = torch.permute(encodedSentenceSUM, (1, 0, 2)) # permuting to have batch at dim 0

        # reducing target dimension, position encoding of target, create target masks and decode
        targetSentencesEmbeddedPaddedBatch_reduced = self.reduceTargetEmbDimNN(targetSentencesEmbeddedPaddedBatch)
        targetSentencesEmbeddedPaddedBatch_reduced_withPosEnc = self.positional_encoding_targetSentence(targetSentencesEmbeddedPaddedBatch_reduced)
        tgt_mask, tgt_padding_mask = create_masks_for_output(targetSentencesEmbeddedPaddedBatch_reduced, whichArePaddingWordsTargetSentenceBatch, self.DEVICE)

        decoderOut = self.decoder(targetSentencesEmbeddedPaddedBatch_reduced_withPosEnc, encodedSentenceSUM, tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_pad_mask)
        
        # from target_emb_size_reduced back to len(finalVoc)
        return self.generator(decoderOut)


    # for the AE task given all sentences in one single tensor similar to what does in forward splits them and computes hidden embedding summing 
    # this is not what the GNN model will use but useful for testing AE
    def encodeSentences(self, inputSentencesAsTensorsSplittedBatch: Tensor, whichArePaddingWordsMaskSplittedBatch: Tensor):
        # list of 1 tensor for each sentence (BATCHDIM, 50, 770)
        sentencesList = [ inputSentencesAsTensorsSplittedBatch[:, i, :, :] for i in range(MAX_NUMBER_OF_SENTENCES_AE) ]
        inputMasksList = [ whichArePaddingWordsMaskSplittedBatch[:, i, :] for i in range(MAX_NUMBER_OF_SENTENCES_AE) ]
        encodedSentenceSUM = []
        for sentence, mask in zip(sentencesList, inputMasksList):
            # create src_mask, key_padding mask
            src_mask, src_padding_mask = create_masks_for_input(sentence, mask, self.DEVICE)
            # add positional encoding
            sentence = self.positional_encoding_sourceSentence(sentence)  
            # pass throw encoder
            encodedSentence = self.encoder(sentence, src_mask, src_padding_mask)
            # now encodedSentence has shape (BATCH_SIZE, MAX_NUM_WORDS, 770) (has has to be input for any other encoder layer when stacking
            encodedSentence_collapsed = torch.mean(encodedSentence, 1) # collapsing dim 1
            encodedSentenceSUM.append(encodedSentence_collapsed)
        encodedSentenceSUM = torch.stack(encodedSentenceSUM).to(self.DEVICE, torch.float32)
        encodedSentenceSUM = torch.permute(encodedSentenceSUM, (1, 0, 2))
        return encodedSentenceSUM


    # to encode single sentence: this is what will be used to get CONCEPT(nodei)
    # remember to work always with BATCH_DIM at first dimension
    def encodeSingleSentence(self, sentenceAsMatrix: Tensor, whichWordsArePadding: Tensor):
        src_mask, src_padding_mask = create_masks_for_input(sentenceAsMatrix, whichWordsArePadding, self.DEVICE)
        # add positional encoding
        sentence = self.positional_encoding_sourceSentence(sentence)  
        # pass throw encoder
        encodedSentence = self.encoder(sentence, src_mask, src_padding_mask)
        # now encodedSentence has shape (BATCH_SIZE, MAX_NUM_WORDS, 770) (has has to be input for any other encoder layer when stacking)
        # we want the output of the encoder module to have dimension (BATCH_SIZE, 770) 
        encodedSentence_collapsed = torch.mean(encodedSentence, 1) # collapsing dim 1
        return encodedSentence_collapsed # this represents sentence representation for the node-sentence, input to GNN


    # decode from 5 node-sentence representations (after GNN) stacked vertically
    def decodeFromEcodedSentenceSUM_afterGNN(self, tgt: Tensor, tgt_mask: Tensor, encodedSentenceSUM_afterGNN: Tensor, memory_key_pad_mask: Tensor):
        tgt_reduced = self.reduceTargetEmbDimNN(tgt)
        tgt_reduced_withPosEnc = self.positional_encoding_targetSentence(tgt_reduced)
        decoderOut = self.decoder(tgt_reduced_withPosEnc, encodedSentenceSUM_afterGNN, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask = memory_key_pad_mask)
        return decoderOut


''' ---------------------------------------------------- TRAINING PREPARATION ----------------------------------------------------'''


''' UNPACK INPUT SENTENCES TO PROVIDE THEM SEPARETELY THROUGH ENCODER '''
# we are working with batched tensors (BATCHDIM x 5*50 x 770)
# in order to provide the sentences to the encoder separately we split dim1 to obtain something like (BATCHDIM x 5 x 50 x 700)
# the 0 in the mask will not allow the transformer to attend to that position
# (grouping by MAX_NUMBER_OF_WORDS_IN_SENTENCE)
def unpack_inputSentencesAsTensorBatch_whichArePaddingWordsMaskBatch(inputSentencesAsTensorBatch, whichArePaddingWordsMaskBatch):
  inputSentencesAsTensorsSplittedBatchList = []
  whichArePaddingWordsMaskSplittedBatchList = []
  # unbatching
  for inputSentencesAsTensor, whichArePaddingWordsMask in zip(inputSentencesAsTensorBatch, whichArePaddingWordsMaskBatch):
    inputSetOfSentencesSplittedList = []
    whichArePaddingWordsMaskSplittedList = []
    for i in range(0, inputSentencesAsTensor.shape[0], MAX_NUMBER_OF_WORDS_IN_SENTENCE):
      inputSentence = inputSentencesAsTensor[i : i+MAX_NUMBER_OF_WORDS_IN_SENTENCE]
      inputSetOfSentencesSplittedList.append(inputSentence)
      inputSentenceMask = whichArePaddingWordsMask[i : i+MAX_NUMBER_OF_WORDS_IN_SENTENCE]
      whichArePaddingWordsMaskSplittedList.append(inputSentenceMask)

    inputSetOfSentencesSplitted = torch.stack(inputSetOfSentencesSplittedList)
    inputSentencesAsTensorsSplittedBatchList.append(inputSetOfSentencesSplitted)

    whichArePaddingWordsMaskSplitted = torch.stack(whichArePaddingWordsMaskSplittedList)
    whichArePaddingWordsMaskSplittedBatchList.append(whichArePaddingWordsMaskSplitted)
  
  # batching back
  inputSentencesAsTensorsSplittedBatch = torch.stack(inputSentencesAsTensorsSplittedBatchList)
  whichArePaddingWordsMaskSplittedBatch = torch.stack(whichArePaddingWordsMaskSplittedBatchList)

  return inputSentencesAsTensorsSplittedBatch, whichArePaddingWordsMaskSplittedBatch


''' IMPLEMENTING EARLY STOPPING '''
class EarlyStopping():
    def __init__(self, patience=10, min_delta=0, mode="min", save_best_weights=True):
        """
        patience : int [default=10]
            Number of epochs without improvement to wait before stopping the training.
        min_delta : float [default=0]
            Minimum value to identify as an improvement.
        mode : str [default="min"]
            One of "min" or "max". Identify whether an improvement
            consists on a smaller or bigger value in the metric.
        save_best_weights : bool [default=True]
            Whether to save the model state when there is an improvement or not.
        """
        # save initialization argument
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        # determine which is the definition of better score given the mode
        if self.mode=="min":
            self._is_better = lambda new, old: new < old - self.min_delta
            self.best_score = np.Inf
        elif self.mode=="max":
            self._is_better = lambda new, old: new > old + self.min_delta
            self.best_score = -np.Inf

        self.save_best_weights = save_best_weights
        self.best_weights = None
        # keep tracks of number of iterations without improvements
        self.n_iter_no_improvements = 0
        # whether to stop the training or not
        self.stop_training = False
        
    def step(self, metric, model=None):
        # if there is an improvements, update `best_score` and save model weights
        if self._is_better(metric, self.best_score):
            self.best_score = metric
            self.n_iter_no_improvements = 0
            if self.save_best_weights and model is not None:
                self.best_weights = model.state_dict()
        # otherwise update counter
        else:
            self.n_iter_no_improvements += 1

        # if no improvements for more epochs than patient, stop the training
        # (set the flag to False)
        if self.n_iter_no_improvements >= self.patience:
            self.stop_training = True
            print(f"Early Stopping: monitored quantity did not improved in the last {self.patience} epochs.")

        return self.stop_training
        
 
''' ---------------------------------------------------- CREATE TRAIN AND SAVE MODEL ----------------------------------------------------'''
def createModel(device):
    # model parameters definition
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    SOURCE_EMB_SIZE = NUMBER_EXPECTED_FEATURES_INPUT #(770)
    NHEAD_ENCODER = 11 # SOURCE_EMB_SIZE // NHEAD_ENCODER = 0 MUST BE DIVISIBLE! 
    TARGET_EMB_SIZE = NUMBER_EXPECTED_FEATURES_OUTPUT # (10136)
    TARGET_EMB_SIZE_REDUCED = NUMBER_EXPECTED_FEATURES_INPUT 
    N_HEAD_DECODER = 11 # TARGET_EMB_SIZE_REDUCED // N_HEAD_DECODER = 0 MUST BE DIVISIBLE!
    DROPOUT = 0.3

    # model instantiation
    transformer = AETransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, SOURCE_EMB_SIZE, NHEAD_ENCODER, TARGET_EMB_SIZE, TARGET_EMB_SIZE_REDUCED, N_HEAD_DECODER, DROPOUT, device)

    # parameters initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

''' ---------------------------------------------------- MULTI GPU TRAINING ----------------------------------------------------'''

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id) # transfer to GPU
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id]) # wrapping model with DDP: to access model need to access self.model.module !!!


    def _run_epoch_train(self, epoch):
        self.train_data.sampler.set_epoch(epoch) # to sample accordingly to epoch number
        self.model.train()
        losses = 0
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        for inputSentencesAsTensorBatch, whichArePaddingWordsMaskBatch, targetSentencesEmbeddedPaddedBatch, whichArePaddingWordsTargetSentenceBatch in self.train_data:
            # unpacking to provide sentences separately through the encoder
            inputSentencesAsTensorsSplittedBatch, whichArePaddingWordsMaskSplittedBatch = unpack_inputSentencesAsTensorBatch_whichArePaddingWordsMaskBatch(inputSentencesAsTensorBatch, whichArePaddingWordsMaskBatch) 
            # converting tensor to DEVICE (to allocate it on GPU)
            memory_key_pad_mask = create_memory_key_pad_mask(whichArePaddingWordsMaskSplittedBatch).to(self.gpu_id)
            inputSentencesAsTensorsSplittedBatch = inputSentencesAsTensorsSplittedBatch.to(self.gpu_id)
            whichArePaddingWordsMaskSplittedBatch = whichArePaddingWordsMaskSplittedBatch.to(self.gpu_id)
            targetSentencesEmbeddedPaddedBatch = targetSentencesEmbeddedPaddedBatch.to(self.gpu_id)
            whichArePaddingWordsTargetSentenceBatch = whichArePaddingWordsTargetSentenceBatch.to(self.gpu_id)
            
            targetSentencesEmbeddedPaddedBatch_input = targetSentencesEmbeddedPaddedBatch[:, :-1, :] # not passing STOP token
            whichArePaddingWordsTargetSentenceBatch_input = whichArePaddingWordsTargetSentenceBatch[:, :-1] # not passing STOP token
            # masks are created inside forward as they need to be created according to each separated sentence
            logits = self.model(inputSentencesAsTensorsSplittedBatch, whichArePaddingWordsMaskSplittedBatch, targetSentencesEmbeddedPaddedBatch_input, whichArePaddingWordsTargetSentenceBatch_input, memory_key_pad_mask)

            self.optimizer.zero_grad() # reset gradients

            targetSentencesEmbeddedPaddedBatch_out = targetSentencesEmbeddedPaddedBatch[:, 1:, :] # not passing START token
            reshaped_tgt_out = targetSentencesEmbeddedPaddedBatch_out.reshape(-1, targetSentencesEmbeddedPaddedBatch_out.shape[-1]) # (250*BATCH_SIZE x len(finalVoc))
            # Now we want to retrieve class index from one-hot vector target (more computationally efficient)
            reshaped_tgt_out_category_label = torch.argmax(reshaped_tgt_out, dim=1) # along dim = 1
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), reshaped_tgt_out_category_label)

            loss.backward() # compute backpropagation
            self.optimizer.step() # update weights according to optimizer

            losses += loss.item()
        
        train_loss = losses / len(self.train_data)
        return train_loss


    def _run_epoch_validation(self, epoch):
        self.validation_data.sampler.set_epoch(epoch) # to sample accordingly to epoch number
        self.model.eval()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        losses = 0
        for inputSentencesAsTensorBatch, whichArePaddingWordsMaskBatch, targetSentencesEmbeddedPaddedBatch, whichArePaddingWordsTargetSentenceBatch in self.validation_data:
            # unpacking to provide sentences separately through the encoder
            inputSentencesAsTensorsSplittedBatch, whichArePaddingWordsMaskSplittedBatch = unpack_inputSentencesAsTensorBatch_whichArePaddingWordsMaskBatch(inputSentencesAsTensorBatch, whichArePaddingWordsMaskBatch) 
            # converting tensor to DEVICE (to allocate it on GPU)
            memory_key_pad_mask = create_memory_key_pad_mask(whichArePaddingWordsMaskSplittedBatch).to(self.gpu_id)
            inputSentencesAsTensorsSplittedBatch = inputSentencesAsTensorsSplittedBatch.to(self.gpu_id)
            whichArePaddingWordsMaskSplittedBatch = whichArePaddingWordsMaskSplittedBatch.to(self.gpu_id)
            targetSentencesEmbeddedPaddedBatch = targetSentencesEmbeddedPaddedBatch.to(self.gpu_id)
            whichArePaddingWordsTargetSentenceBatch = whichArePaddingWordsTargetSentenceBatch.to(self.gpu_id)

            targetSentencesEmbeddedPaddedBatch_input = targetSentencesEmbeddedPaddedBatch[:, :-1, :] # not passing STOP token
            whichArePaddingWordsTargetSentenceBatch_input = whichArePaddingWordsTargetSentenceBatch[:, :-1] # not passing STOP token
            # masks are created inside forward
            logits = self.model(inputSentencesAsTensorsSplittedBatch, whichArePaddingWordsMaskSplittedBatch, targetSentencesEmbeddedPaddedBatch_input, whichArePaddingWordsTargetSentenceBatch_input, memory_key_pad_mask)

            targetSentencesEmbeddedPaddedBatch_out = targetSentencesEmbeddedPaddedBatch[:, 1:, :] # not passing START token
            reshaped_tgt_out = targetSentencesEmbeddedPaddedBatch_out.reshape(-1, targetSentencesEmbeddedPaddedBatch_out.shape[-1]) # (250*BATCH_SIZE x 30136)
            # Now we want to retrieve class index from one-hot vector target (more comp efficient)
            reshaped_tgt_out_category_label = torch.argmax(reshaped_tgt_out, dim=1) # along dim = 1
        
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), reshaped_tgt_out_category_label)
            
            losses += loss.item()

        return losses / len(self.validation_data)
        

    def _save_checkpoint(self, epoch, history):
        ckp = self.model.module.state_dict() # .module to access weiths when module wrapped inside DDP
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': ckp,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, "./src/transfPretrainCP_multiGPU_more_x2_Complex.tar")

        with open('./src/historyTraining_multiGPU_more_x2_Complex.pickle', 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Epoch {epoch} | Training checkpoint saved")


    def train(self, max_epochs: int):
        early_stopping = EarlyStopping(20, save_best_weights=True) # patience=10
        history = { 'train_loss': [], 'validation_loss' : [] }

        early_stopping_occurred = False
        for epoch in range(1, max_epochs+1):
            start_time = timer()

            train_loss = self._run_epoch_train(epoch)
            history['train_loss'].append(train_loss)
            
            val_loss = self._run_epoch_validation(epoch)
            history['validation_loss'].append(val_loss)

            end_time = timer()

            print((f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
            sys.stdout.flush()
            if self.gpu_id == 0 and early_stopping.step(val_loss, self.model.module):
                # if stopping, load best model weights
                self.model.module.load_state_dict(early_stopping.best_weights)
                early_stopping_occurred = True
                break

            # save every 10 epochs checkpoint with model and optimizer states
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, history)

            # save if maxNumberOfEpochs reached or early stopping triggered
        torch.save(self.model.module.state_dict() , "./src/encoderDecoderModelsPretrained_multiGPU_more_x2_Complex.pt")

        with open('./src/historyTraining_more_x2_Complex.pickle', 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return early_stopping_occurred



def load_train_objs(device):
    train_set = PreprocessedDatasetForAE(PATH_TO_DATASET, 'train')  # load your dataset
    validation_set = PreprocessedDatasetForAE(PATH_TO_DATASET, 'validation')
    model = createModel(device) # load your model, pass DEVICE/rank as this will be needed for masks creation on gpu
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    return train_set, validation_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset) # dataset partitioned across all gpus without overlappings
    )


# main process running on each GPU
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_dataset, validation_dataset, model, optimizer = load_train_objs(rank)
    train_data = prepare_dataloader(train_dataset, batch_size)
    validation_data = prepare_dataloader(validation_dataset, batch_size)
    trainer = Trainer(model, train_data, validation_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


# spawns a number of processesses equal to the number of available GPUs 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Model training on multiple GPUs')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=64, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print("Number of GPUs: {}".format(world_size))
    print("Total number of epochs {}".format(args.total_epochs))
    print("Save every: {}".format(args.save_every))
    print("Batch size: {}".format(args.batch_size))
    print(torch.cuda.get_arch_list())
    print(torch.cuda.is_available())
    nltk.data.path.append('./punkt/tokenizers/punkt')
    print("pytorch version")
    print(torch.__version__)
    print("cuda version")
    print(torch.version.cuda)
    print("cuDNN version")
    print(torch.backends.cudnn.version())
    print("arch versions")
    print(torch._C._cuda_getArchFlags())
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)

