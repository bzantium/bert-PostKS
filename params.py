PAD = 0
UNK = 1
SOS = 2
EOS = 3
CLS = 101
SEP = 102

n_vocab = 30522
n_hidden = 768
temperature = 0.8

train_path = "data/train_self_original_no_cands.txt"
test_path = "data/valid_self_original_no_cands.txt"

model_root = "snapshots"
encoder_restore = "snapshots/bert-PostKS-encoder.pt"
Kencoder_restore = "snapshots/bert-PostKS-Kencoder.pt"
manager_restore = "snapshots/bert-PostKS-manager.pt"
decoder_restore = "snapshots/bert-PostKS-decoder.pt"

