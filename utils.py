import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import params
from copy import copy
import torch.backends.cudnn as cudnn
from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def init_model(net, restore=None):

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(), filename)
    print("save pretrained model to: {}".format(filename))


def save_models(model, filenames):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    for i in range(len(model)):
        net = model[0]
        filename = filenames[0]
        torch.save(net.state_dict(), filename)
        print("save pretrained model to: {}".format(filename))


def load_data(path):
    with open(path, errors="ignore") as file:
        X = []
        K = []
        y = []
        k = []

        for line in file:
            dialog_id = line.split()[0]
            if dialog_id == "1":
                k = []

            if "your persona:" in line:
                if len(k) == 3:
                    continue
                k_line = line.split("persona:")[1].strip("\n").lower()
                k.append(k_line)

            elif "__SILENCE__" not in line:
                K.append(k)
                X_line = " ".join(line.split("\t")[0].split()[1:]).lower()
                y_line = line.split("\t")[1].strip("\n").lower()
                X.append(X_line)
                y.append(y_line)

    X_ind = []
    y_ind = []
    K_ind = []

    for line in X:
        tokens = tokenizer.tokenize(line)
        seqs = tokenizer.convert_tokens_to_ids(tokens)
        X_ind.append(seqs)

    for line in y:
        tokens = tokenizer.tokenize(line)
        seqs = tokenizer.convert_tokens_to_ids(tokens)
        y_ind.append(seqs)

    for lines in K:
        K_temp = []
        for line in lines:
            tokens = tokenizer.tokenize(line)
            seqs = tokenizer.convert_tokens_to_ids(tokens)
            K_temp.append(seqs)
        K_ind.append(K_temp)

    return X_ind, y_ind, K_ind


def get_data_loader(X, y, K, n_batch):
    dataset = PersonaDataset(X, y, K)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=n_batch,
        shuffle=True
    )
    return data_loader


class PersonaDataset(Dataset):
    def __init__(self, X, y, K):
        X_len = max([len(tokens) for tokens in X]) + 2
        y_len = max([len(tokens) for tokens in y]) + 1
        k_len = 0
        for lines in K:
            for line in lines:
                if k_len < len(line) + 2:
                    k_len = len(line) + 2

        src_X = list()
        src_y = list()
        src_K = list()
        tgt_y = list()

        for tokens in X:
            tokens.insert(0, params.CLS)
            tokens.append(params.SEP)
            tokens.extend([params.PAD] * (X_len - len(tokens)))
            src_X.append(tokens)

        for tokens in y:
            src_line = copy(tokens)
            tgt_line = copy(tokens)
            src_line.insert(0, params.SOS)
            tgt_line.append(params.EOS)
            src_line.extend([params.PAD] * (y_len - len(src_line)))
            tgt_line.extend([params.PAD] * (y_len - len(tgt_line)))
            src_y.append(src_line)
            tgt_y.append(tgt_line)

        for N_tokens in K:
            src_k = list()
            for tokens in N_tokens:
                tokens.insert(0, params.CLS)
                tokens.append(params.SEP)
                tokens.extend([params.PAD] * (k_len - len(tokens)))
                src_k.append(tokens)
            src_K.append(src_k)

        self.src_X = torch.LongTensor(src_X)
        self.src_y = torch.LongTensor(src_y)
        self.src_K = torch.LongTensor(src_K)
        self.tgt_y = torch.LongTensor(tgt_y)
        self.dataset_size = len(self.src_X)

    def __getitem__(self, index):
        src_X = self.src_X[index]
        src_y = self.src_y[index]
        tgt_y = self.tgt_y[index]
        src_K = self.src_K[index]
        return src_X, src_y, src_K, tgt_y

    def __len__(self):
        return self.dataset_size


def knowledgeToIndex(K):
    k1, k2, k3 = K

    tokens = tokenizer.tokenize(k1)
    K1 = tokenizer.convert_tokens_to_ids(tokens)

    tokens = tokenizer.tokenize(k2)
    K2 = tokenizer.convert_tokens_to_ids(tokens)

    tokens = tokenizer.tokenize(k3)
    K3 = tokenizer.convert_tokens_to_ids(tokens)

    K = [K1, K2, K3]
    seq_len = max([len(k) for k in K]) + 2

    K1.insert(0, params.CLS)
    K1.append(params.SEP)
    K2.insert(0, params.CLS)
    K2.append(params.SEP)
    K3.insert(0, params.CLS)
    K3.append(params.SEP)

    K1.extend([params.PAD] * (seq_len - len(K1)))
    K2.extend([params.PAD] * (seq_len - len(K2)))
    K3.extend([params.PAD] * (seq_len - len(K3)))

    K1 = torch.LongTensor(K1).unsqueeze(0)
    K2 = torch.LongTensor(K2).unsqueeze(0)
    K3 = torch.LongTensor(K3).unsqueeze(0)
    K = torch.cat((K1, K2, K3), dim=0).unsqueeze(0).cuda()  # K: [1, 3, seq_len]
    return K