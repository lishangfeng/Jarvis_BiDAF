#!/usr/bin/enc python3
# BiDAF model of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Highway import Highway
import logging

logger = logging.getLogger(__name__)

class BiDAF_model(nn.Module):
    # nerual netword of BiDAF
    def __init__(self, args, encoding_layer, attentionflow_layer, model_layer, output_layer):
        super(BiDAF_model, self).__init__()
        self.encoding_layer = encoding_layer
        self.attentionflow_layer = attentionflow_layer
        self.model_layer = model_layer
        self.output_layer = output_layer

    def forward(self, context_info, context_feature, context_mask, query_info, query_mask, context_char, query_char, toBePrint = False):

        context_encoding, query_encoding = self.encoding_layer(context_info, context_feature, context_mask, query_info, query_mask, context_char, query_char)
        attentionflow = self.attentionflow_layer(context_encoding, context_mask, query_encoding, query_mask)
        model_info = self.model_layer(attentionflow)
        start_prob, end_prob = self.output_layer(attentionflow, model_info, context_mask)

        return start_prob, end_prob

class Encoding_Layer(nn.Module):
    """
    Encoding layer of BiDAF
    """
    def __init__(self, args, word_dict, char_dict):
        super(Encoding_Layer, self).__init__()
        self.args = args
        self.word_dict = word_dict
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim) #, padding_idx = 0)
        if args.use_char_emb:
            self.embedding_char = nn.Embedding(len(char_dict), args.char_emb_dim )# , padding_idx = 0)
            # self.embedding_char.weight.data.normal_(0, 0.5)
            self.conv = nn.Conv2d(args.char_emb_dim, args.out_char_dim, kernel_size = (1, args.cnn_kernel_size))
            highway_in_dim = args.embedding_dim + args.out_char_dim
        else:
            highway_in_dim = args.embedding_dim
        #     self.linear_context = nn.Linear(args.embedding_dim + args.char_emb_dim, args.model_dim)
        #     self.linear_query = nn.Linear(args.embedding_dim + args.char_emb_dim, args.model_dim)
        # else:
        #     self.linear_context = nn.Linear(args.embedding_dim, args.model_dim)
        #     self.linear_query = nn.Linear(args.embedding_dim, args.model_dim)        
        self.highway_context = Highway(highway_in_dim, args.num_highways)
        # self.highway_query = Highway(highway_in_dim, args.num_highways)
        if args.add_features:
            self.LSTM_context = nn.LSTM(input_size = highway_in_dim + args.num_features, hidden_size = args.model_dim,
                                       num_layers = 1, bidirectional = True)
        else:
            self.LSTM_context = nn.LSTM(input_size = highway_in_dim, hidden_size = args.model_dim,
                                      num_layers = 1, bidirectional = True)
        if not args.share_lstm_weight:
            self.LSTM_query = nn.LSTM(input_size = highway_in_dim, hidden_size = args.model_dim,
                                      num_layers = 1, bidirectional = True)
        if args.embedding_file:
            self.load_embedding(word_dict, args.embedding_file)


    def forward(self, context, context_feature, context_mask, query, query_mask, context_char, query_char):
        """
        Inputs:
            context: document word indices [batch * len_d]
            context_feature: document word feature indices [batch * len_d * nfeat]
            context_mask: document padding mask
            query: question word indices
            query_mask: question padding mask
        """

        # embedding both context and query
        context_info = self.embedding(context)
        query_info = self.embedding(query)
        
        if self.args.use_char_emb:
            context_char_emb = self.charater_embedding(context_char)
            query_char_emb = self.charater_embedding(query_char)
            
            context_info = torch.cat((context_info, context_char_emb), dim = 2)
            query_info = torch.cat((query_info, query_char_emb), dim = 2)

        context_info = self.highway_context(context_info)
        query_info = self.highway_context(query_info)
        # query_info = self.highway_query(query_info)
        # context_info = self.highway_context(self.embedding(context))
        # query_info = self.highway_query(self.embedding(query))
        # context_info = self.linear_context(self.highway_context(self.embedding(context)))
        # query_info = self.linear_query(self.highway_query(self.embedding(query)))
        
        # Add manual features
        if self.args.add_features:
            context_info = torch.cat((context_info, context_feature), 2)
        
        # B x T x C -> T x B x C
        context_info = context_info.transpose(0, 1)
        query_info = query_info.transpose(0, 1) 

        context_info = F.dropout(context_info, p = self.args.dropout, training = self.training)
        query_info = F.dropout(query_info, p = self.args.dropout, training = self.training)
        
        # Encoding the information
        context_info, _ = self.LSTM_context(context_info)
        if self.args.share_lstm_weight:
            query_info, _ = self.LSTM_context(query_info)
        else:
            query_info, _ = self.LSTM_query(query_info)
        # T x B x C -> B x T x C
        context_info = context_info.transpose(0, 1)
        query_info = query_info.transpose(0, 1) 

        # Masking the output
        context_mask = context_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 2)
        query_mask = query_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 2)
        
        context_info.data.masked_fill_(context_mask, 0)
        query_info.data.masked_fill_(query_mask, 0)

        return context_info, query_info

    def load_embedding(self, word_dict, embedding_file):
        """
        Loading pretrained embeddings for a given list of word, if they exist.
        Args:
            words: iterable of tokens. Only those that are indexed in the 
                dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in word_dict.tokens()}
        logger.info("Loading pre-trained embeddingd for %d words from %s" % 
            (len(words), embedding_file))
        embedding = self.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embendding)
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)
        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' % 
            (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def charater_embedding(self, chars):
        char_emb = self.embedding_char(chars)
        char_emb = F.dropout(char_emb, p = self.args.dropout, training = self.training)
        char_emb = char_emb.permute(0, 3, 1, 2)
        char_emb = self.conv(char_emb)
        char_emb = F.relu(char_emb.permute(0, 2, 3, 1))
        char_emb = torch.max(char_emb, 2)[0].squeeze(2)
        return char_emb


class AttentionFlow_Layer(nn.Module):
    """
    Attention class in BiDAF, according to the paper.
    """
    def __init__(self, args):
        super(AttentionFlow_Layer, self).__init__()
        self.args = args
        self.linear_similarity_matrix = nn.Linear(args.model_dim * 6, 1)
        self.linear_inner = nn.Linear(args.model_dim * 8, args.model_dim * 8)
        self.linear_output = nn.Linear(args.model_dim * 8, args.model_dim * 8)

    def forward(self, context_info, context_mask, query_info, query_mask):
        similarity_matrix = self.get_similarity_matrix(context_info, context_mask, query_info, query_mask)

        c2q_attention = self.get_c2q_attention(similarity_matrix, query_info)
        q2c_attention = self.get_q2c_attention(similarity_matrix, context_info)

        result = torch.cat((context_info, c2q_attention, context_info * c2q_attention, context_info * q2c_attention), 2)
        result = self.linear_output(F.relu(self.linear_inner(result)))
        context_mask_output = context_mask.unsqueeze(2).repeat(1, 1, self.args.model_dim * 8)
        result.data.masked_fill_(context_mask_output, 0)
        # B x T x C    C = model_dim * 8
        return result

    def get_similarity_matrix(self, context_info, context_mask, query_info, query_mask):

        tiled_context_info = context_info.unsqueeze(2).expand(context_info.size()[0],
                                                              context_info.size()[1],
                                                              query_info.size()[1],
                                                              context_info.size()[2]
                                                              )
        tiled_query_info = query_info.unsqueeze(1).expand(query_info.size()[0],
                                                          context_info.size()[1],
                                                          query_info.size()[1],
                                                          query_info.size()[2],
                                                          )
        tiled_context_mask = context_mask.unsqueeze(2).expand(context_mask.size()[0],
                                                              context_mask.size()[1],
                                                              query_info.size()[1],
                                                              )
        tiled_query_mask = query_mask.unsqueeze(1).expand(query_mask.size()[0],
                                                          context_mask.size()[1],
                                                          query_mask.size()[1],
                                                          )
        # Get the attention mask
        attn_mask = torch.ge(tiled_context_mask + tiled_query_mask, 1)
        
        cross_info = tiled_context_info * tiled_query_info

        concat_info = torch.cat((tiled_context_info, tiled_query_info, cross_info), 3)

        # Get the high dimentional mask
        attn_mask_concat = attn_mask.unsqueeze(3).repeat(1, 1, 1, self.args.model_dim * 6)

        # Mask the concat_info
        concat_info.data.masked_fill_(attn_mask_concat, 0)

        similarity_matrix = self.linear_similarity_matrix(concat_info).squeeze(3)

        # Mask the final result
        similarity_matrix.data.masked_fill_(attn_mask, -1e10)
        
        # B x Tc x Tq
        return similarity_matrix

    def get_c2q_attention(self, similarity_matrix, query_info):
    
        c2q_similarity_matrix = F.softmax(similarity_matrix, dim = -1)
        c2q_attention = torch.bmm(c2q_similarity_matrix, query_info)

        # B x Tc x C   C = model_dim * 2
        return c2q_attention 

    def get_q2c_attention(self, similarity_matrix, context_info):
        _similarity_matrix = torch.max(similarity_matrix, dim = 2)[0]
        q2c_similarity_matrix = F.softmax(_similarity_matrix, dim = 1)
        q2c_similarity_matrix = q2c_similarity_matrix.unsqueeze(1)
        
        q2c_attention = torch.bmm(q2c_similarity_matrix, context_info)
        q2c_attention = q2c_attention.repeat(1, context_info.size(1), 1)

        # B x Tc x C    C = model_dim * 2
        return q2c_attention 

class Model_layer(nn.Module):
    """
    Modeling Layer, encodes the query-aware representations of context words.
    """
    def __init__(self, args):
        self.args = args
        super(Model_layer, self).__init__()
        self.LSTMs = nn.ModuleList()
        self.LSTMs.append(nn.LSTM(input_size = args.model_dim * 8, hidden_size = args.model_dim, 
                           num_layers = 1, bidirectional = True))
        for i in range(args.model_lstm_layers-1):
            self.LSTMs.append(nn.LSTM(input_size = args.model_dim * 2, hidden_size = args.model_dim, 
                           num_layers = 1, bidirectional = True))

    def forward(self, query_aware_representation):
        inp = query_aware_representation.transpose(0, 1)
        
        for i in range(self.args.model_lstm_layers):
            inp = F.dropout(inp, p = self.args.dropout, training = self.training)
            inp, _ = self.LSTMs[i](inp)
        output = inp.transpose(0, 1)
        return output

class Output_layer(nn.Module):
    """
    Output Layer to predict the span.
    """
    def __init__(self, args):
        super(Output_layer, self).__init__()
        self.args = args
        self.linear_start = nn.Linear(args.model_dim * 10, 1)
        self.linear_end = nn.Linear(args.model_dim * 10, 1)
        self.LSTMs = nn.ModuleList()
        for i in range(args.output_lstm_layers):
            self.LSTMs.append(nn.LSTM(input_size = args.model_dim * 2, hidden_size = args.model_dim, 
                              num_layers = 1, bidirectional = True))

    def forward(self, Attention_output, Model_output, context_mask):
        start_inp = torch.cat((Attention_output, Model_output), 2)
        start_inp = F.dropout(start_inp, p = self.args.dropout, training = self.training)
        start_prob = self.linear_start(start_inp).squeeze(2)
        # start_prob = F.dropout(start_prob, p = self.args.dropout, training = self.training)
        start_prob.data.masked_fill_(context_mask, -1e08)
        
        # Model_output = F.dropout(Model_output, p = self.args.dropout, training = self.training)
        # Model_output, _ = self.RNN(Model_output)
        Model_output = Model_output.transpose(0, 1)
        for i in range(self.args.output_lstm_layers):
            Model_output = F.dropout(Model_output, p = self.args.dropout, training = self.training)
            Model_output, _ = self.LSTMs[i](Model_output)
        Model_output = Model_output.transpose(0, 1)

        end_inp = torch.cat((Attention_output, Model_output), 2)
        end_inp = F.dropout(end_inp, p = self.args.dropout, training = self.training)
        end_prob = self.linear_end(end_inp).squeeze(2)
        # end_prob = F.dropout(end_prob, p = self.args.dropout, training = self.training)
        end_prob.data.masked_fill_(context_mask, -1e08)
        
        if self.training:
            start_prob = F.log_softmax(start_prob, dim = -1)
            end_prob = F.log_softmax(end_prob, dim = -1)
        else:
            start_prob = F.softmax(start_prob, dim = -1)
            end_prob = F.softmax(end_prob, dim = -1)
        
        return start_prob, end_prob

def build_model(args, word_dict, char_dict, normalize):
    encoding_layer = Encoding_Layer(args, word_dict, char_dict)
    attentionflow_layer = AttentionFlow_Layer(args)
    model_layer = Model_layer(args)
    output_layer = Output_layer(args)
    return BiDAF_model(args, encoding_layer, attentionflow_layer, model_layer, output_layer)


