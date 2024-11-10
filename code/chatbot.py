import training as train_mod
import main

# Initialize search module
encoder = main.encoder
decoder = main.encoder
voc = main.voc
searcher = train_mod.GreedySearchDecoder(encoder, decoder)

train_mod.evaluateInput(encoder, decoder, searcher, voc)