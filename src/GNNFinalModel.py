'''

- Between Extractive & Abstractive Document Summarization Project 
- Andrew Zamai & Alessandro Manfè
- LFN course 2022/23, Prof. Fabio Vandin

GNN FINAL MODEL (multi GPUs training on BladeCluster DEI)
Freezed transformer layers, LSTM like message propagation

'''

# libraries
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from timeit import default_timer as timer
import pickle
import math
import os
import nltk
import copy
import sys
import argparse

# PyG
from torch.nn import Module
from torch_geometric.data import Data, Batch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, Sequential
from torch.nn import Linear, Parameter, ReLU
from torch_geometric.utils import degree

# our defined library of functions
import libraryForBuildingDatasetOptimized as l4bdOptimized

# for multi-GPUs training
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


''' ------------------------------------------ DistributedDataParallelSetup ------------------------------------------ '''

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process # assigned automatically 
        world_size: Total number of processes # 1 process on each GPU --> will be number of GPUs available
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


''' ------------------------------------------ HERE WE DEFINE SOME GLOBAL VARIABLES ------------------------------------------- '''

BATCH_SIZE = 32 # adjust according to GPU RAM available
EARLY_STOPPING = 200

PATH_TO_DATASET= '/nfsd/VFdisk/zamaiandre/DocSumProject/data/GraphsDataset' # path to folder containing articles preprocesses as graphs
PATH_TO_CSV_OF_TOPENGLISHFREQWORDS = '/nfsd/VFdisk/zamaiandre/DocSumProject/data/topEnglishWords.csv' # csv with top frequent English words

PATH_TO_TRANSFORMER_CHECKPOINT = '/nfsd/VFdisk/zamaiandre/DocSumProject/src/final_more_x2_complex_ae.tar' # pretrained encoder and decoder weights

fileName = __file__.split('/')[-1][:-3]
PATH_TO_GNN_CHECKPOINT = os.path.join('/nfsd/VFdisk/zamaiandre/DocSumProject/src', fileName + '_checkpoint.tar')
PATH_TO_HISTORY = os.path.join('/nfsd/VFdisk/zamaiandre/DocSumProject/src', fileName + '_history.pickle')
PATH_TO_STATE_DICT = os.path.join('/nfsd/VFdisk/zamaiandre/DocSumProject/src', fileName + '_finalModel.pt')

# these 2 variables MUST be same of the ones used to preprocess the articles to produce PreprocessedDataset
FINAL_VOCABULARY_SIZE = 9998 # special characters, UNK tags and tokens like [START] [EOS] exluded --> will be ca 10136 (NUMBER_EXPECTED_FEATURES_OUTPUT)
MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS = 100 # make sure number of UNK tags is equal to the one with which the articles were preprocessed 

MAX_NUMBER_OF_WORDS_IN_SENTENCE = 50 # a sentence in input will be padded or trunked to this fixed length (required by transformer architecture)
MAX_NUMBER_OF_SENTENCES_AE = 5 # number of sentences fed separately in input to the decoder and required in output by the decoder

NUMBER_EXPECTED_FEATURES_INPUT = 770 # 768 from BERT embedding + 2 added by us for UNK tags
NUMBER_EXPECTED_FEATURES_OUTPUT = FINAL_VOCABULARY_SIZE + 138 # (+138 takes into account UNKi tags, special tokens and special characters added when building FinalVocabulary) = 10136


''' ----------------------------------------------------- HELPER FUNCTIONS ----------------------------------------------------- '''

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


'''--------------------------------------------------DATASET PREPARATION------------------------------------------------------'''

class PreprocessedDatasetForGNN(Dataset):
    def __init__(self, pathToFolder, trainValidationOrTest):
        super().__init__()
        self._indices = None
        self.pathToFolder = os.path.join(pathToFolder, trainValidationOrTest) # eg './data/GraphsDataset' , 'train'
        self.finalVocabularyAsDict = l4bdOptimized.loadTargetVocabulary(FINAL_VOCABULARY_SIZE, MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS, path=PATH_TO_CSV_OF_TOPENGLISHFREQWORDS) 
        self.maxNumberOfWordsInSentence = MAX_NUMBER_OF_WORDS_IN_SENTENCE
        self.maxNumberOfSentences = MAX_NUMBER_OF_SENTENCES_AE
        self.trainValidationOrTest = trainValidationOrTest
        
        nltk.data.path.append('/nfsd/VFdisk/zamaiandre/DocSumProject/punkt')
        nltk.data.path.append('/nfsd/VFdisk/zamaiandre/DocSumProject/punkt/tokenizers/punkt')

    def len(self):
        if self.trainValidationOrTest == 'validation':
            return len(os.listdir(self.pathToFolder))
        else:
            return len(os.listdir(self.pathToFolder)) # if I don't want to use all training samples 

    def get(self, idx):
        data = None
        pathToArticle = os.path.join(self.pathToFolder, str(idx) + '.pickle')
        with open(pathToArticle, 'rb') as handle:
            data = pickle.load(handle)

        # unpacking the pickle
        graphData = data["graphData"]
        encodedSentences = graphData["nodeEmbedding"]
        edgeIndex = graphData["edgeIndex"]
        edgeWeight = graphData["edgeWeight"]
        highlights = data["highlights"]
        unknownWordsDictionaryForAnArticle = data["UnknownVocabulary"]

        # sometime highlights are as list, other times not: we unpack them
        if isinstance(highlights,(tuple,list)):
            highlights = highlights[0]
        # same check for unknownWordsDictionaryForAnArticle
        if len(unknownWordsDictionaryForAnArticle)>0 and isinstance(list(unknownWordsDictionaryForAnArticle.values())[0],(list,tuple)):
            for k in unknownWordsDictionaryForAnArticle:
                unknownWordsDictionaryForAnArticle[k] = unknownWordsDictionaryForAnArticle[k][0]
        
        # computing embedding for target highlights on the fly as they are heavy in memory but fast to compute
        # NB: unknownWordsDictionaryForAnArticle has to be the same produced when preprocessing source article
        targetSentencesEmbedded = l4bdOptimized.preprocessHighlights(highlights, unknownWordsDictionaryForAnArticle, self.finalVocabularyAsDict)

        # to enable batching and speed up computation we need to have them all to same dimension
        # the target is obtained by concatenating all sentences and embedding them in a single matrix of dimension (#words, finalVocabularyLen)
        # we need to pad along dim=0, we may chose as dim0 = #self.maxNumberOfWordsInSentence * self.maxNumberOfSentences
        # we need to have like a mask [1, 1, 1, 1, 0] so we can understand which are padding elements and which are not
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
        
        return Data(x=encodedSentences.type(torch.float32),
                    edge_index=edgeIndex.type(torch.int), 
                    edge_attr=edgeWeight.type(torch.float32), 
                    ohe_highlights=targetSentencesEmbeddedPadded.type(torch.float32), 
                    target_padding=whichArePaddingWordsTargetSentence.type(torch.int))
        

'''--------------------------------------------------- MODULES DEFINITION ------------------------------------------------------'''

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


''' TRANFORMER MODEL DEFINITION:
Remember: input sentences are passed individually 1 by 1, their condensed representations stacked at the bottleneck and fed all together to the decoder.
NB: since the encoder works stackin N encoder layers the output dim of the encoder will be (source_emb_size x 50 x 770)(has to be the same as the input to be fed potentially into another encoder layer)
From matrix representing a sentence (50 x 770) to a single vector embedding but of same "width" (770) we need to perform some manipulation (eg 'mean' to collapse 1st dimension)
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
        encodedSentenceSUM = torch.permute(encodedSentenceSUM, (1, 0, 2)) # permuting to have batch at dim 0
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


    # decode from 5 node-sentence representations (after GNN) stacked vertically, used for Greedy Decoding method
    def decodeFromEcodedSentenceSUM_afterGNN(self, tgt: Tensor, tgt_mask: Tensor, encodedSentenceSUM_afterGNN: Tensor, memory_key_pad_mask: Tensor):
        tgt_reduced = self.reduceTargetEmbDimNN(tgt)
        tgt_reduced_withPosEnc = self.positional_encoding_targetSentence(tgt_reduced)
        decoderOut = self.decoder(tgt_reduced_withPosEnc, encodedSentenceSUM_afterGNN, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask = memory_key_pad_mask)
        return decoderOut


''' 
GNN MODEL DEFINITION (MESSAGES PASSING PART):
Message passing layer of the core gnn submodel
The message creation mimics what a LSTM cell do in RNNs.
A NN works on the concatenation of the parent embedding and child embedding deciding what to add (or subtract if negative)
A NN acts as gate modulating what to add of the output of the above NN.
'''
class CondenseConcept(MessagePassing):
  def __init__(self, device, in_channels, out_channels):
    super().__init__(aggr='add')

    self.linFaucet = nn.Sequential(
        Linear(2*in_channels, 770, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        Linear(770, 385, bias=True),
        nn.ReLU(),
        Linear(385, out_channels, bias=True)
    )

    self.linAdd = nn.Sequential(
        Linear(2*in_channels, 770, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        Linear(770, 385, bias=True),
        nn.ReLU(),
        Linear(385, out_channels, bias=True)
    )

    self.linSelf = nn.Sequential(
        Linear(2*in_channels, 1155, bias=True),
        nn.ReLU(),
        Linear(1155, out_channels, bias=True)
    )
    
    self.bias = Parameter(torch.Tensor(out_channels))

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.dropout = nn.Dropout(p=0.2)

    self.device = device

    self.reset_parameters()

  def reset_parameters(self):
    # self.linSelf.reset_parameters()
    self.bias.data.zero_()
  
  def forward(self, x, edge_index, edge_weight, iter=1):
    '''
    x: shape (num_nodes for all graphs in batch, in_channels) # batching works constructing a big adjacency matrix and not adding a batch dimension
    edge_index: shape (2, total num_edges) # all edges --> will be used to compute the big adjacency matrix
    edge_weight: shape (total num_edges)
    '''

    node_from, node_to = edge_index
    deg_to = degree(node_to, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = (deg_to).pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[node_to] # deg_inv_sqrt[node_from] * # only indegree
    norm = norm.to(self.device)

    out = x
    out = out.to(self.device)

    #iterations cycle
    for i in range(iter):
        out = self.propagate(edge_index=edge_index, edge_weight=edge_weight, norm=norm, x=out)
    return out
    
  def message(self, x_i, x_j, edge_weight, norm):
    # x_i features node "to" ht-1
    # x_j features nodes "from" xt

    concatenated = torch.cat((x_i, x_j), dim=1).to(self.device) # (E, 1540)
    concatenated = self.dropout(concatenated)
    forgetAndAdd = (self.sigmoid(self.linFaucet(concatenated)) * (self.linAdd(concatenated)) ) 

    return forgetAndAdd # * norm.view(-1,1) 
    
  def update(self, aggr_out, x):
    '''
    aggr_out: shape (batch_size, num_nodes, out_channels)
    x: shape (batch_size, num_nodes, in_channels)
    '''
    concatenated = torch.cat((x, aggr_out), dim=1).to(self.device)
    result = self.relu( self.linSelf(concatenated) + self.bias )

    return result


''' 
GNN MODEL DEFINITION: ( NODE SELECTION PART ):
Masking Layer for the top k meaningful nodes according to in-degree
'''
class TopKNodes(Module):
    def __init__(self, k, device):
        super().__init__()
        self.k = k
        self.device = device
  
    def forward(self, x, edge_index):
        kk = min(self.k, x.size(0)) # is the number of nodes in the graph > k ?
        _, node_to = edge_index
        deg = degree(node_to, x.size(0), dtype=x.dtype) # computing indegree for each node

        indices = torch.topk(deg, kk).indices.to(self.device) # retrieving to kk indices of the nodes with higher indegree
        padding = torch.zeros((max(self.k - x.size(0), 0), NUMBER_EXPECTED_FEATURES_INPUT)).to(self.device) # padding with 0s if number of nodes < k (memory_key_padding mask will be computed in the GNNDecoder forward pass)
        return torch.cat((x[indices], padding), dim = 0)


''' FULL GNN MODEL DEFINITION: '''

class GNNDecoder(nn.Module):

    def __init__(self, device, params, encoder=None, decoder=None, reducer=None, generator=None):
        super().__init__()
        self.DEVICE = device
        # initializing the GNN custom layer  
        self.GNNmodel = CondenseConcept(self.DEVICE, params["SOURCE_EMB_SIZE"], params["SOURCE_EMB_SIZE"])
        # top k nodes module selector
        self.TopKLayer = TopKNodes(MAX_NUMBER_OF_SENTENCES_AE, self.DEVICE)
        # loading encoder and decoder weights
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=params["SOURCE_EMB_SIZE"], 
                                                                        nhead=params["NHEAD_ENCODER"],
                                                                        dim_feedforward=512,
                                                                        dropout=0.2,
                                                                        activation='relu',
                                                                        layer_norm_eps = 1e-5,
                                                                        batch_first = True,
                                                                        norm_first = False),
                                                                        num_layers=params["NUM_ENCODER_LAYERS"])
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=params["TARGET_EMB_SIZE_REDUCED"],
                                                                        nhead=params["N_HEAD_DECODER"],
                                                                        dim_feedforward=512,
                                                                        dropout=0.2,
                                                                        activation='relu',
                                                                        layer_norm_eps = 1e-5,
                                                                        batch_first = True,
                                                                        norm_first = False),
                                                                        num_layers=params["NUM_DECODER_LAYERS"])
        if reducer is not None:
            self.reducer = reducer
        else:
            self.reduceTargetEmbDimNN = nn.Sequential(nn.Linear(params["TARGET_EMB_SIZE"], params["TARGET_EMB_SIZE_REDUCED"]), nn.ReLU())
        if generator is not None:
            self.generator = generator
        else:
            self.generator = nn.Linear(params["TARGET_EMB_SIZE_REDUCED"], params["TARGET_EMB_SIZE"])

        # positional_encoding_sourceSentence works on a single sentence representation at the time of length MAX_NUMBER_OF_WORDS_IN_SENTENCE
        self.positional_encoding_sourceSentence = PositionalEncoding(params["SOURCE_EMB_SIZE"], 0.2, MAX_NUMBER_OF_WORDS_IN_SENTENCE)
        # positional_encoding_targetSentence works on entire target sentence (obtained concatenating the sentences in input)
        self.positional_encoding_targetSentence = PositionalEncoding(params["TARGET_EMB_SIZE_REDUCED"], 0.2, MAX_NUMBER_OF_WORDS_IN_SENTENCE * MAX_NUMBER_OF_SENTENCES_AE)

        self.dropoutOnEncodedSentSUM = nn.Dropout(p=0.2)


    def forward(self, batch):

        ITERATION = 1 # number of iterations 
        # passing data in batched form through GNN
        condensedSentences = self.GNNmodel(batch.x.to(self.DEVICE), batch.edge_index.type(torch.int64).to(self.DEVICE), batch.edge_attr.to(self.DEVICE), ITERATION)
        nodeMap = batch.batch.to(self.DEVICE) # maps each node to its graph (batching works constructing a big adjacency matrix of all graphs)
        
        encodedSentenceSUM = []
        memory_key_pad_mask_list = []
    
        # the transformer works in a different batching mode, thus an unwrapping and repacking is needed 
        for i in range(len(batch)):
            graphNodes = condensedSentences[nodeMap == i] # retrieving back graph nodes according to nodeMap
            edge_index = batch.get_example(i).edge_index.type(torch.int64)
            topKNodes = self.TopKLayer(graphNodes, edge_index) # finding top k nodes
            encodedSentenceSUM.append(topKNodes)
            # if number of nodes < k (=MAX_NUMBER_OF_SENTENCES_AE for now) we need to compute memory_key_padding mask
            if graphNodes.size(0) < MAX_NUMBER_OF_SENTENCES_AE:
                mask = [0 if i<graphNodes.size(0) else 1 for i in range(MAX_NUMBER_OF_SENTENCES_AE)] # False if no padding, True if padding
                memory_key_pad_mask_list.append(torch.tensor(mask, dtype=torch.bool))
            else:
                memory_key_pad_mask_list.append(torch.zeros(MAX_NUMBER_OF_SENTENCES_AE, dtype=torch.bool)) # all False valid 

        encodedSentenceSUM = (torch.stack(encodedSentenceSUM, dim=0)).to(self.DEVICE, torch.float32)
        memory_key_pad_mask = torch.stack(memory_key_pad_mask_list).to(self.DEVICE)

        # dropout on encodedSentenceSUM
        encodedSentenceSUM = self.dropoutOnEncodedSentSUM(encodedSentenceSUM)

        # target sentences to pass to decoder
        targetSentencesEmbeddedPaddedBatch = torch.reshape(batch.ohe_highlights, (-1, MAX_NUMBER_OF_SENTENCES_AE * MAX_NUMBER_OF_WORDS_IN_SENTENCE, NUMBER_EXPECTED_FEATURES_OUTPUT))
        whichArePaddingWordsTargetSentenceBatch = torch.reshape(batch.target_padding, (-1, MAX_NUMBER_OF_SENTENCES_AE * MAX_NUMBER_OF_WORDS_IN_SENTENCE))

        targetSentencesEmbeddedPaddedBatch = targetSentencesEmbeddedPaddedBatch[:, :-1, :].to(self.DEVICE) # not passing STOP token
        whichArePaddingWordsTargetSentenceBatch = whichArePaddingWordsTargetSentenceBatch[:, :-1].to(self.DEVICE) # not passing STOP token
        
        # reducing target dimension, position encoding of target, create target masks and decode
        targetSentencesEmbeddedPaddedBatch_reduced = self.reducer(targetSentencesEmbeddedPaddedBatch)
        targetSentencesEmbeddedPaddedBatch_reduced_withPosEnc = self.positional_encoding_targetSentence(targetSentencesEmbeddedPaddedBatch_reduced)
        tgt_mask, tgt_padding_mask = create_masks_for_output(targetSentencesEmbeddedPaddedBatch_reduced, whichArePaddingWordsTargetSentenceBatch, self.DEVICE)

        decoderOut = self.decoder(targetSentencesEmbeddedPaddedBatch_reduced_withPosEnc, encodedSentenceSUM, tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_pad_mask)
        
        # from target_emb_size_reduced back to len(finalVoc)
        return self.generator(decoderOut)


''' ---------------------------------------------------- CREATE AND LOAD MODEL ----------------------------------------------------'''


def createModel(device):
    # model parameters definition, have to be the same of the transformer encoder and decoder pretrained model
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    SOURCE_EMB_SIZE = NUMBER_EXPECTED_FEATURES_INPUT #(770)
    NHEAD_ENCODER = 11 # SOURCE_EMB_SIZE // NHEAD_ENCODER = 0 MUST BE DIVISIBLE! 
    TARGET_EMB_SIZE = NUMBER_EXPECTED_FEATURES_OUTPUT # (10136)
    TARGET_EMB_SIZE_REDUCED = NUMBER_EXPECTED_FEATURES_INPUT 
    N_HEAD_DECODER = 11 # TARGET_EMB_SIZE_REDUCED // N_HEAD_DECODER = 0 MUST BE DIVISIBLE!
    DROPOUT = 0.3

    transformerDefinedParams = {"NUM_ENCODER_LAYERS": NUM_ENCODER_LAYERS,
                                "NUM_DECODER_LAYERS": NUM_DECODER_LAYERS,
                                "SOURCE_EMB_SIZE": NUMBER_EXPECTED_FEATURES_INPUT,
                                "NHEAD_ENCODER": 11,
                                "TARGET_EMB_SIZE": NUMBER_EXPECTED_FEATURES_OUTPUT,
                                "TARGET_EMB_SIZE_REDUCED": NUMBER_EXPECTED_FEATURES_INPUT,
                                "N_HEAD_DECODER": 11,
                                "DROPOUT": 0.3}

    # model instantiation
    transformer = AETransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, SOURCE_EMB_SIZE, NHEAD_ENCODER, TARGET_EMB_SIZE, TARGET_EMB_SIZE_REDUCED, N_HEAD_DECODER, DROPOUT, device)
    # loading pretrained weights
    checkpoint = torch.load(PATH_TO_TRANSFORMER_CHECKPOINT, map_location= ('cuda:' + str(device)))
    transformer.load_state_dict(checkpoint['model_state_dict'])

    # creating a deepcopy of these weights to use them in the GNNDecoder complete model
    encoderCopy = copy.deepcopy(transformer.encoder)
    decoderCopy = copy.deepcopy(transformer.decoder)
    reducerCopy = copy.deepcopy(transformer.reduceTargetEmbDimNN)
    generatorCopy = copy.deepcopy(transformer.generator)

    model = GNNDecoder(params = transformerDefinedParams, device=device, encoder=encoderCopy, decoder=decoderCopy, reducer=reducerCopy, generator=generatorCopy)
    
    # when training the GNNDecoder model we freeze the decoder, reducer, and generator weights to force the GNN to train first
    # at some point in the train we will unfreeze them
    print("Modules in GNNDecoder: ")
    for name, child in model.named_children():
        print(name)
        if name in ['encoder', 'decoder', 'reducer', 'generator']:
            child.requires_grad_(False)
    print("Trainable parameters: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            
    model = model.to(device)

    sys.stdout.flush()

    return model


'''-------------------------------------------------------- TRAIN MODEL -----------------------------------------------------------'''

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


'''TRAINER DEFINITION:'''

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id) # transfer to GPU
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True) # wrapping model with DDP: to access model need to access self.model.module !!!
        self.batch_size = batch_size

    def _run_epoch_train(self, epoch, lossClassWeights):

        self.train_data.sampler.set_epoch(epoch) # to sample accordingly to epoch number

        self.model.train()

        losses = 0
        # weighting more UNK tag classes
        loss_fn = torch.nn.CrossEntropyLoss(weight=lossClassWeights, ignore_index=0)
        
        for batch in self.train_data:
            # batch is a batched version of Data containing (encodedSentences, edgeIndex, edgeWeight, targetSentencesEmbeddedPaddedBatch, whichArePaddingWordsTargetSentenceBatch)
            logits = self.model(batch)

            self.optimizer.zero_grad() # reset gradients

            targetSentencesEmbeddedPaddedBatch = torch.reshape(batch.ohe_highlights.to(self.gpu_id), (-1, MAX_NUMBER_OF_SENTENCES_AE * MAX_NUMBER_OF_WORDS_IN_SENTENCE, NUMBER_EXPECTED_FEATURES_OUTPUT))
            targetSentencesEmbeddedPaddedBatch_out = targetSentencesEmbeddedPaddedBatch[:, 1:, :] # not passing START token
            reshaped_tgt_out = targetSentencesEmbeddedPaddedBatch_out.reshape(-1, targetSentencesEmbeddedPaddedBatch_out.shape[-1]) # (250*BATCH_SIZE x len(finalVoc))
            # Now we want to retrieve class index from one-hot vector target (more computationally efficient)
            reshaped_tgt_out_category_label = torch.argmax(reshaped_tgt_out, dim=1) # along dim = 1
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), reshaped_tgt_out_category_label)

            loss.backward() # compute backpropagation
            self.optimizer.step() # update weights according to optimizer

            losses += loss.item()
        
        return losses / len(self.train_data)


    def _run_epoch_validation(self, epoch, lossClassWeights):

        self.validation_data.sampler.set_epoch(epoch) # to sample accordingly to epoch number

        self.model.eval()

        loss_fn = torch.nn.CrossEntropyLoss(weight=lossClassWeights, ignore_index=0)
        losses = 0
       
        for batch in self.validation_data:

            logits = self.model(batch)

            targetSentencesEmbeddedPaddedBatch = torch.reshape(batch.ohe_highlights.to(self.gpu_id), (-1, MAX_NUMBER_OF_SENTENCES_AE * MAX_NUMBER_OF_WORDS_IN_SENTENCE, NUMBER_EXPECTED_FEATURES_OUTPUT))

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
                    }, PATH_TO_GNN_CHECKPOINT)

        with open(PATH_TO_HISTORY, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Epoch {epoch} | Training checkpoint saved")


    
    def train(self, startingEpoch, max_epochs: int):

        early_stopping = EarlyStopping(EARLY_STOPPING, save_best_weights=True)
        
        # only 1 gpu saving history
        if self.gpu_id == 0:
            history = { 'train_loss': [], 'validation_loss' : [] }

        #    history = None
        #    with open(PATH_TO_HISTORY, 'rb') as handle:
        #        history = pickle.load(handle)

        finalVocabularyAsDict = l4bdOptimized.loadTargetVocabulary(FINAL_VOCABULARY_SIZE, MAX_NUMBER_OF_TAGS_FOR_UNK_WORDS, path=PATH_TO_CSV_OF_TOPENGLISHFREQWORDS)
        lossClassWeights = torch.Tensor( [ 2 if len(str(x))>3 and str(x)[:4] == '[UNK' else 1 for x in finalVocabularyAsDict.keys()] ).to(self.gpu_id)

        early_stopping_occurred = False
        for epoch in range(startingEpoch, max_epochs+1):
            start_time = timer()

            train_loss = self._run_epoch_train(epoch, lossClassWeights)
            
            val_loss = self._run_epoch_validation(epoch, lossClassWeights)
            
            if self.gpu_id == 0:
                history['train_loss'].append(train_loss)
                history['validation_loss'].append(val_loss)

            end_time = timer()

            print((f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
            sys.stdout.flush()

            if self.gpu_id == 0 and early_stopping.step(val_loss, self.model.module):
                # if stopping, load best model weights
                self.model.module.load_state_dict(early_stopping.best_weights)
                early_stopping_occurred = True
                break

            # save every save_every epochs checkpoint with model and optimizer states
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, history)

        # save if maxNumberOfEpochs reached or early stopping triggered
        if self.gpu_id == 0:
            torch.save(self.model.module.state_dict() , PATH_TO_STATE_DICT)
            with open(PATH_TO_HISTORY, 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return early_stopping_occurred


def load_train_objs(device):
    train_set = PreprocessedDatasetForGNN(PATH_TO_DATASET, 'train')  # load your dataset
    validation_set = PreprocessedDatasetForGNN(PATH_TO_DATASET, 'validation')
    model = createModel(device) # load your model, pass DEVICE/rank as this will be needed for masks creation on gpu
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    return train_set, validation_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        sampler=DistributedSampler(dataset) # dataset partitioned across all gpus without overlappings
    )

'''-------------------------------------------------------- SINGLE GPU MAIN and MAIN spawning processes -----------------------------------------------------------'''

# on each GPU
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_dataset, validation_dataset, model, optimizer = load_train_objs(rank)
    train_data = prepare_dataloader(train_dataset, batch_size)
    validation_data = prepare_dataloader(validation_dataset, batch_size)
    trainer = Trainer(model, train_data, validation_data, optimizer, rank, save_every, batch_size)
    startingEpoch = 0
    trainer.train(startingEpoch, total_epochs)
    destroy_process_group()


# spawns a number of processesses equal to the number of available GPUs 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training on multiple GPUs')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a checkpoint (model and optimizer)')
    parser.add_argument('--batch_size', default=BATCH_SIZE, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print("Number of GPUs: {}".format(world_size))

    print("Total number of epochs {}".format(args.total_epochs))

    print("Save every: {}".format(args.save_every))

    print("Batch size: {}".format(args.batch_size))

    print(torch.cuda.get_arch_list())
    print(torch.cuda.is_available())

    nltk.data.path.append('/nfsd/VFdisk/zamaiandre/DocSumProject/punkt/tokenizers/punkt')

    print("pytorch version")
    print(torch.__version__)
    print("cuda version")
    print(torch.version.cuda)
    print("cuDNN version")
    print(torch.backends.cudnn.version())
    print("arch versions")
    print(torch._C._cuda_getArchFlags())

    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
