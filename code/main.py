import os
import random
import torch
import torch.nn as nn
from torch import optim

import preprocessing as prep_mod
import training as train_mod

def main():
    # preprocessing
    txt_path = os.path.join(os.path.abspath('data'), 'dialogues_text2.txt')
    print(txt_path)
    raw_lines = prep_mod.read_in(txt_path)
    cleaned_lines = list(map(prep_mod.clean_up, raw_lines))
    cleaned_lines=list(filter(lambda x: x and not x.isupper() and x[0]!='(', cleaned_lines))

    MAX_LENGTH = 10
    save_dir = os.path.abspath('model')
    voc, pairs = prep_mod.loadPrepareData(cleaned_lines, MAX_LENGTH)

    MIN_COUNT = 5
    pairs = prep_mod.trimRareWords(voc, pairs, MIN_COUNT)
    
    # initializing model
    model_name = 'chat_model'
    attn_model = 'dot'
    hidden_size = 600
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.3
    batch_size = 64
    checkpoint_iter = 4000
    model_path = os.path.join(os.path.abspath('model'), model_name)
    loadFilename = os.path.join(model_path,
                    '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                    '{}_checkpoint.tar'.format(checkpoint_iter))
    print("Searching for model at: ", loadFilename)

    # Configure training/optimization
    clip = 2.0
    teacher_forcing_ratio = 0.7
    learning_rate = 0.0001
    decoder_learning_ratio = 0.5
    n_iteration = 4000
    print_every = 10
    save_every = 500
    
    # Load an existing model if it exists
    if os.path.exists(loadFilename)==True:
        print("Loading model...")
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

        # Initialize word embeddings
        embedding = nn.Embedding(voc.num_words, hidden_size)
        embedding.load_state_dict(embedding_sd)
        # Initialize encoder & decoder models
        encoder = train_mod.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = train_mod.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

        # Use appropriate device
        encoder = encoder.to(train_mod.device)
        decoder = decoder.to(train_mod.device)

        # Initialize optimizers
        # encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay = 0.001)
        # decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, weight_decay = 0.001)
        # encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        # decoder_optimizer.load_state_dict(decoder_optimizer_sd)      
    else: 
        print("Building new model...")
        # Initialize word embeddings
        embedding = nn.Embedding(voc.num_words, hidden_size)

        # Initialize encoder & decoder models
        encoder = train_mod.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = train_mod.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

        # Use appropriate device
        encoder = encoder.to(train_mod.device)
        decoder = decoder.to(train_mod.device)

        # Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()

        # Initialize optimizers
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay = 0.001)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, weight_decay = 0.001)

        # If you have CUDA, configure CUDA to call
        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # Run training iterations
        print("Starting Training!")
        train_mod.trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, encoder_n_layers, decoder_n_layers, hidden_size, save_dir, n_iteration, batch_size,
                print_every, save_every, clip, teacher_forcing_ratio, MAX_LENGTH, loadFilename)
        
        print('Models built and ready to go!')
    # Set dropout layers to ``eval`` mode
    encoder.eval()
    decoder.eval()

    searcher = train_mod.GreedySearchDecoder(encoder, decoder)
    print("Ready to chat!! Type 'quit' or 'q' to stop chatting")
    train_mod.evaluateInput(encoder, decoder, searcher, voc, MAX_LENGTH)
    print("Bye!!")

    return

main()
