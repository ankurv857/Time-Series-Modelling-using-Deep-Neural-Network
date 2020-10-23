import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict
import os
from os.path import join
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from Config.arguments import get_args ; args = get_args()

from Model_utils.MultiEmbedding import MultiEmbedding 
from Model_utils.Perceptron import Perceptron

class Neural_Network_lstmfused(nn.Module):
    """module containing 1 rnn 1 mlp they respective encoders with int embedding layers and a decoder"""
    def __init__(self, emb_tmpl_list_inp, emb_tmpl_list_out, emb_static_list_inp, emb_static_list_out, mlp_tmpl_id, mlp_static_id):
        super().__init__()

        # Initialize the embedding layers
        self.embed_temporal = MultiEmbedding((emb_tmpl_list_inp), (emb_tmpl_list_out), sequenced=True)  # Temporal embedding
        self.embed_static = MultiEmbedding((emb_static_list_inp), (emb_static_list_out), sequenced=False)  # Static embedding

        #Hidden Layers
        in_enc_stat_dim = mlp_static_id + sum(emb_static_list_out) ; out_enc_stat_dim = 2**(int(np.math.log(mlp_static_id + sum(emb_static_list_inp),2))) #; print('in_enc_stat_dim' , in_enc_stat_dim)
        in_rnn_dim = mlp_tmpl_id + sum(emb_tmpl_list_out) ; out_rnn_dim = 2**(int(np.math.log(mlp_tmpl_id + sum(emb_tmpl_list_out),2))) #; print('in_enc_temp_dim' , in_enc_temp_dim)
        in_dec_dim = out_enc_stat_dim + out_rnn_dim  ; out_dec_dim = 2**(int(np.math.log(out_enc_stat_dim + out_rnn_dim ,2)))//4
        in_dec_final_dim = out_dec_dim ; out_dec_final_dim = 1 

        #Initialize the static encoder
        self.encoder_static = nn.Linear(in_enc_stat_dim , out_enc_stat_dim) 

        # Initialize  LSTM #Input: seq_length x batch_size x input_size
        self.rnn = nn.LSTM(in_rnn_dim, out_rnn_dim, args.rnn_layers) 
        # Initialize  decoder
        self.decoder = nn.Linear(in_dec_dim , out_dec_dim) 
        self.num_fc_layers = np.math.log(out_dec_dim,2)//4 ; out_dec_interim_dim = out_dec_dim  ; in_dim = [] ; out_dim = [] ;self.decoder_interim = []
        for i in range(int(self.num_fc_layers)):
            in_dim.append(out_dec_interim_dim) ; out_dim.append(out_dec_interim_dim//4) #; print('in_dim' , in_dim , 'out_dim' , out_dim)
            decoder_interim = nn.Linear(in_dim[i] , out_dim[i]) #; print('decoder' , decoder_interim)
            self.decoder_interim.append(decoder_interim) #; print('decoder_interim' , self.decoder_interim) wert
            out_dec_interim_dim = out_dim[i]
        self.decoder_final = nn.Linear(out_dec_interim_dim , out_dec_final_dim)  

    def forward(self, mlp_static_data, mlp_tmpl_data , emb_static_data, emb_tmpl_data):
        #Get the static data in 2 Dimension
        emb_static_data = emb_static_data.view(emb_static_data.size()[0] * emb_static_data.size()[1], emb_static_data.size()[2]) #; print('emb_static_data' , emb_static_data.size() , emb_static_data)
        mlp_static_data = mlp_static_data.view(mlp_static_data.size()[0] * mlp_static_data.size()[1], mlp_static_data.size()[2]) #; print('mlp_static_data' , mlp_static_data.size())

        emb_temporal = self.embed_temporal(emb_tmpl_data) #; print('emb_temporal' , emb_temporal , emb_temporal.size())
        emb_static = self.embed_static(emb_static_data) #; print('emb_static' , emb_static , emb_static.size())

        # enc_data = torch.cat((mlp_tmpl_data, mlp_static_data), dim=2)
        enc_data = torch.cat((mlp_tmpl_data, emb_temporal), dim=2) ; enc_data_size = enc_data.size()
        enc_data = enc_data.view(enc_data_size[1] , enc_data_size[0] , -1) #; print('enc_data' , enc_data.size()) #; torch.Size([6, 1, 14])

        rnn_out, c_n = self.rnn(enc_data) ; rnn_sizes = rnn_out.size() #; print('rnn_out'  , rnn_out.size())
        mlp_enc = F.relu(self.encoder_static(torch.cat((mlp_static_data,emb_static), dim=1))) ; mlp_sizes = mlp_enc.size() #; print('mlp_sizes'  , mlp_sizes)

        #flattens rnn_out to batch_size*seq_len,rnn_out_size
        rnn_out = rnn_out.contiguous().view(rnn_sizes[0] * rnn_sizes[1], rnn_sizes[2]) #; print('rnn_out' , rnn_out.size() , 'mlp_out' , mlp_out.size())
        decoder = F.relu(self.decoder(torch.cat((rnn_out, mlp_enc), dim=1))) ; decoder_interim = decoder #;  print('decoder_interim1' , decoder_interim.size())  
        for i in range(int(self.num_fc_layers)):
            decoder_interim = F.relu(self.decoder_interim[i](decoder_interim)) #; print('decoder_interim' , decoder_interim.size()) werwre
        preds = F.sigmoid(self.decoder_final(decoder_interim))  
        preds = preds.view(-1, 2*args.seq_len) #; print('preds' , preds.size())
        return preds


class Neural_Network_wavenet(nn.Module):
    def __init__(self, emb_tmpl_list_inp, emb_tmpl_list_out, emb_static_list_inp, emb_static_list_out, mlp_tmpl_id, mlp_static_id):
        super().__init__()

        # Initialize the embedding layers
        self.embed_temporal = MultiEmbedding((emb_tmpl_list_inp), (emb_tmpl_list_out), sequenced=True)  # Temporal embedding
        self.embed_static = MultiEmbedding((emb_static_list_inp), (emb_static_list_out), sequenced=False)  # Static embedding

        in_enc_stat_dim = mlp_static_id + sum(emb_static_list_out) ; out_enc_stat_dim = 2**(int(np.math.log(mlp_static_id + sum(emb_static_list_inp),2))) #; print('in_enc_stat_dim' , in_enc_stat_dim)
        in_cnn1_dim = mlp_tmpl_id + sum(emb_tmpl_list_out) ; out_cnn1_dim = 2**(int(np.math.log(mlp_tmpl_id + sum(emb_tmpl_list_out),2))) #; print('in_enc_temp_dim' , in_enc_temp_dim)
        kernel_size_conv1 = 2 ; dilation_conv1 = 2
        in_cnn2_dim = out_cnn1_dim ; out_cnn2_dim = 2*args.seq_len

        in_dec_dim = out_enc_stat_dim + 2*args.seq_len  ; out_dec_dim = 2**(int(np.math.log(out_enc_stat_dim + out_cnn2_dim ,2)))//4
        in_dec_final_dim = out_dec_dim ; out_dec_final_dim = 1

        # self.dense_layer_input = (self.n_hidden//4) * (2*args.seq_len - self.kernel_size_conv1 - self.kernel_size_conv2 - self.dilation_conv1 - self.dilation_conv1 + 4) 
        
        #Initialize the static encoder
        self.encoder_static = nn.Linear(in_enc_stat_dim , out_enc_stat_dim) 

        #Initialize the conv layers
        self.conv1= torch.nn.Conv1d(in_cnn1_dim, out_cnn1_dim ,kernel_size = kernel_size_conv1 , dilation= dilation_conv1) 
        self.conv2= torch.nn.Conv1d(in_cnn2_dim,out_cnn2_dim, kernel_size= 1 ) 
        self.conv_transpose = nn.ConvTranspose1d(out_cnn2_dim, out_cnn2_dim  ,kernel_size = kernel_size_conv1 , dilation= dilation_conv1 ) 

        # Initialize  decoder
        self.decoder = nn.Linear(in_dec_dim , out_dec_dim) 
        self.num_fc_layers = np.math.log(out_dec_dim,2)//4 ; out_dec_interim_dim = out_dec_dim  ; in_dim = [] ; out_dim = [] ;self.decoder_interim = []
        for i in range(int(self.num_fc_layers)):
            in_dim.append(out_dec_interim_dim) ; out_dim.append(out_dec_interim_dim//4) #; print('in_dim' , in_dim , 'out_dim' , out_dim)
            decoder_interim = nn.Linear(in_dim[i] , out_dim[i]) #; print('decoder' , decoder_interim) sdgfdgf
            self.decoder_interim.append(decoder_interim) #; print('decoder_interim' , self.decoder_interim)
            out_dec_interim_dim = out_dim[i]
        self.decoder_final = nn.Linear(out_dec_interim_dim , out_dec_final_dim)  

    def forward(self, mlp_static_data, mlp_tmpl_data , emb_static_data, emb_tmpl_data):
        
        #Get the static data in 2 Dimension
        emb_static_data = emb_static_data.view(emb_static_data.size()[0] * emb_static_data.size()[1], emb_static_data.size()[2])
        mlp_static_data = mlp_static_data.view(mlp_static_data.size()[0] * mlp_static_data.size()[1], mlp_static_data.size()[2])

        emb_temporal = self.embed_temporal(emb_tmpl_data) #; print('emb' , emb , emb.size())
        emb_static = self.embed_static(emb_static_data) #; print('emb' , emb , emb.size())

        #Encoder over the static data
        mlp_enc = F.relu(self.encoder_static(torch.cat((mlp_static_data,emb_static), dim=1))) ; mlp_sizes = mlp_enc.size() #; print('mlp_sizes'  , mlp_sizes)

        #CNNs over the temporal data
        enc_data = torch.cat((mlp_tmpl_data, emb_temporal), dim=2) ; enc_data_size = enc_data.size()
        #(batch_size , channels , sequence)
        enc_data = enc_data.view(-1  , enc_data_size[2] , enc_data_size[1] ) #; print('enc_data1'  , enc_data.size()) # enc_data1 torch.Size([1, 14, 6])
        conv1 = F.relu(self.conv1(enc_data)) #; print('conv1' , conv1.size()) #torch.Size([1, 8, 4])
        conv2 = F.relu(self.conv2(conv1)) #; print('conv2' , conv2.size()) #torch.Size([25, 6, 4])
        conv_transpose = F.relu(self.conv_transpose(conv2)) #; print('conv_transpose' , conv_transpose.size()) #torch.Size([25, 6, 4])
        conv_transpose = conv_transpose.view(conv_transpose.size()[0]*(conv_transpose.size()[2]) , conv_transpose.size()[1]) #;print('conv_transpose' , conv_transpose.size()) #torch.Size([1, 8])
        
        decoder = F.relu(self.decoder(torch.cat((conv_transpose, mlp_enc), dim=1))) ; decoder_interim = decoder #;  print('decoder' , decoder.size()) 
        for i in range(int(self.num_fc_layers)):
            decoder_interim = F.relu(self.decoder_interim[i](decoder_interim)) #; print('decoder_interim' , decoder_interim.size())
        preds = F.sigmoid(self.decoder_final(decoder_interim)) #; print('preds' , preds.size())  
        preds = preds.view(-1, 2*args.seq_len)  #; print('preds' , preds.size())  
        return preds


class Neural_Network_cnnlstm(nn.Module):
    def __init__(self, emb_tmpl_list_inp, emb_tmpl_list_out, emb_static_list_inp, emb_static_list_out, mlp_tmpl_id, mlp_static_id):
        super().__init__()

        # Initialize the embedding layers
        self.embed_temporal = MultiEmbedding((emb_tmpl_list_inp), (emb_tmpl_list_out), sequenced=True)  # Temporal embedding
        self.embed_static = MultiEmbedding((emb_static_list_inp), (emb_static_list_out), sequenced=False)  # Static embedding

        in_enc_stat_dim = mlp_static_id + sum(emb_static_list_out) ; out_enc_stat_dim = 2**(int(np.math.log(mlp_static_id + sum(emb_static_list_inp),2))) #; print('in_enc_stat_dim' , in_enc_stat_dim)
        in_cnn1_dim = mlp_tmpl_id + sum(emb_tmpl_list_out) ; out_cnn1_dim = 2**(int(np.math.log(mlp_tmpl_id + sum(emb_tmpl_list_out),2))) #; print('in_enc_temp_dim' , in_enc_temp_dim)
        kernel_size_conv1 = 2 ; dilation_conv1 = 2
        in_cnn2_dim = out_cnn1_dim ; out_cnn2_dim = 2*args.seq_len
        in_rnn_dim = mlp_tmpl_id + sum(emb_tmpl_list_out) ; out_rnn_dim = 2**(int(np.math.log(mlp_tmpl_id + sum(emb_tmpl_list_out),2)))


        in_dec_dim = out_enc_stat_dim + 2*args.seq_len + out_rnn_dim ; out_dec_dim = 2**(int(np.math.log(out_enc_stat_dim + out_cnn2_dim ,2)))//4
        in_dec_final_dim = out_dec_dim ; out_dec_final_dim = 1

        # self.dense_layer_input = (self.n_hidden//4) * (2*args.seq_len - self.kernel_size_conv1 - self.kernel_size_conv2 - self.dilation_conv1 - self.dilation_conv1 + 4) 
        
        #Initialize the static encoder
        self.encoder_static = nn.Linear(in_enc_stat_dim , out_enc_stat_dim) 

        #Initialize the conv layers
        self.conv1= torch.nn.Conv1d(in_cnn1_dim, out_cnn1_dim ,kernel_size = kernel_size_conv1 , dilation= dilation_conv1) 
        self.conv2= torch.nn.Conv1d(in_cnn2_dim,out_cnn2_dim, kernel_size= 1 ) 
        self.conv_transpose = nn.ConvTranspose1d(out_cnn2_dim, out_cnn2_dim  ,kernel_size = kernel_size_conv1 , dilation= dilation_conv1 ) 

        # Initialize  LSTM #Input: seq_length x batch_size x input_size
        self.rnn = nn.LSTM(in_rnn_dim, out_rnn_dim, args.rnn_layers) 

        # Initialize  decoder
        self.decoder = nn.Linear(in_dec_dim , out_dec_dim) 
        self.num_fc_layers = np.math.log(out_dec_dim,2)//4 ; out_dec_interim_dim = out_dec_dim  ; in_dim = [] ; out_dim = [] ;self.decoder_interim = []
        for i in range(int(self.num_fc_layers)):
            in_dim.append(out_dec_interim_dim) ; out_dim.append(out_dec_interim_dim//4) #; print('in_dim' , in_dim , 'out_dim' , out_dim)
            decoder_interim = nn.Linear(in_dim[i] , out_dim[i]) #; print('decoder' , decoder_interim)
            self.decoder_interim.append(decoder_interim) #; print('decoder_interim' , self.decoder_interim)
            out_dec_interim_dim = out_dim[i]
        self.decoder_final = nn.Linear(out_dec_interim_dim , out_dec_final_dim) 

    def forward(self, mlp_static_data, mlp_tmpl_data , emb_static_data, emb_tmpl_data):
        
        #Get the static data in 2 Dimension
        emb_static_data = emb_static_data.view(emb_static_data.size()[0] * emb_static_data.size()[1], emb_static_data.size()[2])
        mlp_static_data = mlp_static_data.view(mlp_static_data.size()[0] * mlp_static_data.size()[1], mlp_static_data.size()[2])

        emb_temporal = self.embed_temporal(emb_tmpl_data) #; print('emb' , emb , emb.size())
        emb_static = self.embed_static(emb_static_data) #; print('emb' , emb , emb.size())

        #Encoder over the static data
        mlp_enc = F.relu(self.encoder_static(torch.cat((mlp_static_data,emb_static), dim=1))) #; mlp_sizes = mlp_enc.size() ; print('mlp_sizes'  , mlp_sizes)

        #CNNs over the temporal data
        enc_data = torch.cat((mlp_tmpl_data, emb_temporal), dim=2) ; enc_data_size = enc_data.size()
        #(batch_size , channels , sequence)
        enc_data = enc_data.view(-1  , enc_data_size[2] , enc_data_size[1] ) #; print('enc_data1'  , enc_data.size()) # enc_data1 torch.Size([1, 14, 6])
        conv1 = F.relu(self.conv1(enc_data)) #; print('conv1' , conv1.size()) #torch.Size([1, 8, 4])
        conv2 = F.relu(self.conv2(conv1)) #; print('conv2' , conv2.size()) #torch.Size([1, 2, 4])
        conv_transpose = F.relu(self.conv_transpose(conv2)) 
        conv_transpose = conv_transpose.view(conv_transpose.size()[0]*(conv_transpose.size()[2]) , conv_transpose.size()[1]) 


        # LSTM network #Input: seq_length x batch_size x input_size
        enc_data_lstm = torch.cat((mlp_tmpl_data, emb_temporal), dim=2) ; enc_data_lstm_size = enc_data_lstm.size()
        enc_data_lstm = enc_data_lstm.view(enc_data_lstm_size[1] , enc_data_lstm_size[0] , -1) #; print('enc_data_lstm' , enc_data_lstm.size()) #; torch.Size([6, 1, 14])

        rnn_out, c_n = self.rnn(enc_data_lstm) ; rnn_sizes = rnn_out.size() ; rnn_out = rnn_out.contiguous().view(rnn_sizes[0] * rnn_sizes[1], rnn_sizes[2]) #; print('rnn_out'  , rnn_out.size())
        
        decoder = F.relu(self.decoder(torch.cat((conv_transpose,rnn_out ,mlp_enc), dim=1)))  ; decoder_interim = decoder #;  print('decoder' , decoder.size()) 
        for i in range(int(self.num_fc_layers)):
            decoder_interim = F.relu(self.decoder_interim[i](decoder_interim)) #;  print('decoder' , decoder.size())  
        preds = F.sigmoid(self.decoder_final(decoder_interim))  #; print('preds' , preds.size())  
        preds = preds.view(-1, 2*args.seq_len)  #; print('preds' , preds.size())  
        return preds

class Neural_Network_mlp(nn.Module):
    def __init__(self, emb_tmpl_list_inp, emb_tmpl_list_out, emb_static_list_inp, emb_static_list_out, mlp_tmpl_id, mlp_static_id):
        super().__init__()

        # Initialize the embedding layers
        self.embed = MultiEmbedding((emb_tmpl_list_inp + emb_static_list_inp), (emb_tmpl_list_out + emb_static_list_out), sequenced=False) 

        #Hidden Layers
        in_dec_dim = mlp_tmpl_id + mlp_static_id + sum(emb_tmpl_list_out + emb_static_list_out) 
        out_dec_dim = 2**(int(np.math.log(mlp_tmpl_id + mlp_static_id + sum(emb_tmpl_list_out + emb_static_list_out),2)))
        out_dec_final_dim = 1
        
        # Initialize  decoder
        self.decoder = nn.Linear(in_dec_dim , out_dec_dim) 
        self.num_fc_layers = np.math.log(out_dec_dim,2)//4 ; out_dec_interim_dim = out_dec_dim  ; in_dim = [] ; out_dim = [] ;self.decoder_interim = []
        for i in range(int(self.num_fc_layers)):
            in_dim.append(out_dec_interim_dim) ; out_dim.append(out_dec_interim_dim//4) #; print('in_dim' , in_dim , 'out_dim' , out_dim)
            decoder_interim = nn.Linear(in_dim[i] , out_dim[i]) #; print('decoder' , decoder_interim)
            self.decoder_interim.append(decoder_interim) #; print('decoder_interim' , self.decoder_interim)
            out_dec_interim_dim = out_dim[i]
        self.decoder_final = nn.Linear(out_dec_interim_dim , out_dec_final_dim) 

    def forward(self, mlp_static_data, mlp_tmpl_data , emb_static_data, emb_tmpl_data):
        emb_tmpl_data = emb_tmpl_data.view(emb_tmpl_data.size()[0] * emb_tmpl_data.size()[1], emb_tmpl_data.size()[2])
        emb_static_data = emb_static_data.view(emb_static_data.size()[0] * emb_static_data.size()[1], emb_static_data.size()[2])
        mlp_static_data = mlp_static_data.view(mlp_static_data.size()[0] * mlp_static_data.size()[1], mlp_static_data.size()[2])
        mlp_tmpl_data = mlp_tmpl_data.view(mlp_tmpl_data.size()[0] * mlp_tmpl_data.size()[1], mlp_tmpl_data.size()[2])

        #print('emb_tmpl_data, emb_static_data' , emb_tmpl_data.size(), emb_static_data.size())
        emb_data = torch.cat((emb_tmpl_data, emb_static_data), dim=1) 
        emb = self.embed(emb_data) #; print('emb' , emb , emb.size())

        enc_data = torch.cat((mlp_tmpl_data, mlp_static_data), dim=1)
        mlp_emb_concat = torch.cat((enc_data, emb), dim=1) 

        decoder = F.relu(self.decoder(mlp_emb_concat))  ; decoder_interim = decoder #;  print('decoder' , decoder.size()) 
        for i in range(int(self.num_fc_layers)):
            decoder_interim = F.relu(self.decoder_interim[i](decoder_interim)) #;  print('decoder' , decoder.size())  asd
        preds = F.sigmoid(self.decoder_final(decoder_interim))  #; print('preds' , preds.size())  
        preds = preds.view(-1, 2*args.seq_len)  #; print('preds' , preds.size())  
        return preds


class Neural_Network_lstmcell(nn.Module):
    def __init__(self, emb_tmpl_list_inp, emb_tmpl_list_out, emb_static_list_inp, emb_static_list_out, mlp_tmpl_id, mlp_static_id):
        super().__init__()
        self.embed = MultiEmbedding( (emb_tmpl_list_inp + emb_static_list_inp), (emb_tmpl_list_out + emb_static_list_out), sequenced=True)  # Temporal embedding
        lstm1_inp_dim = mlp_tmpl_id + mlp_static_id + sum(emb_tmpl_list_out + emb_static_list_out)
        hidden_size = 2**(int(np.math.log(mlp_tmpl_id + mlp_static_id + sum(emb_tmpl_list_out + emb_static_list_out),2))) #; print('hidden_size', hidden_size)

        self.lstm1 = nn.LSTMCell(lstm1_inp_dim, hidden_size) 
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size//4) 
        self.linear1 = nn.Linear(hidden_size//4, hidden_size//8)
        self.linear2 = nn.Linear(hidden_size//8, 1)
        self.hidden_size = hidden_size

    def forward(self, mlp_static_data, mlp_tmpl_data , emb_static_data, emb_tmpl_data , future= 0):
        emb_data = torch.cat((emb_tmpl_data, emb_static_data), dim=2) 
        emb = self.embed(emb_data)
        enc_data = torch.cat((mlp_tmpl_data, mlp_static_data), dim=2)
        enc_data = torch.cat((enc_data, emb), dim=2) ; enc_data_size = enc_data.size()
        enc_data = enc_data.view(enc_data_size[1] , enc_data_size[0] , -1) #; print('enc_data1'  , enc_data.size()) #enc_data1 torch.Size([10, 1, 76])
        input = enc_data #; print('input' , input.size()) ; torch.Size([1, 10, 75])

        outputs = []
        h_t = Variable(torch.FloatTensor(input.size(0), self.hidden_size).zero_())
        c_t = Variable(torch.FloatTensor(input.size(0), self.hidden_size).zero_()) #; print('ht ct' , h_t.size() , c_t.size())
        h_t2 = Variable(torch.FloatTensor(input.size(0), self.hidden_size//4).zero_())
        c_t2 = Variable(torch.FloatTensor(input.size(0), self.hidden_size//4).zero_())
        
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = input_t.squeeze(dim=1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            linear_1 = F.relu(self.linear1(h_t2))
            output = F.sigmoid(self.linear2(linear_1))
            outputs += [output] 
        outputs = torch.stack(outputs, 1).squeeze(2) ; outputs = outputs.view(outputs.size()[1] ,outputs.size()[0]  )
        return outputs
