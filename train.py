import random
import argparse
import torch
from torch import optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import params
from utils import init_model, save_model, \
    load_data, get_data_loader
from model import Encoder, KnowledgeEncoder, Decoder, Manager


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-pre_epoch', type=int, default=5,
                   help='number of epochs for pre_train')
    p.add_argument('-n_epoch', type=int, default=15,
                   help='number of epochs for train')
    p.add_argument('-n_batch', type=int, default=32,
                   help='number of batches for train')
    p.add_argument('-lr', type=float, default=5e-4,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    p.add_argument('-tfr', type=float, default=0.8,
                   help='teacher forcing ratio')
    p.add_argument('-restore', default=False, action='store_true',
                   help='whether restore trained model')
    return p.parse_args()


def pre_train(model, optimizer, train_loader, args):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.train(), Kencoder.train(), manager.train(), decoder.train()
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)

    for epoch in range(args.pre_epoch):
        for step, (src_X, src_y, src_K, _) in enumerate(train_loader):
            src_X = src_X.cuda()
            src_y = src_y.cuda()
            src_K = src_K.cuda()

            optimizer.zero_grad()
            _, x = encoder(src_X)
            n_batch = src_y.size(0)
            CLS_tokens = torch.LongTensor([params.CLS] * n_batch).unsqueeze(1).cuda()
            SEP_tokens = torch.LongTensor([params.SEP] * n_batch).unsqueeze(1).cuda()
            src_y_ = torch.cat((CLS_tokens, src_y[:, 1:], SEP_tokens), dim=-1)
            y = Kencoder(src_y_)
            K = Kencoder(src_K)

            n_vocab = decoder.n_vocab
            seq_len = src_y.size(1) - 1
            _, _, _, k_logits = manager(x, y, K) # k_logits: [n_batch, n_vocab]
            k_logits = k_logits.repeat(seq_len, 1, 1).transpose(0, 1).contiguous().view(-1, n_vocab)
            bow_loss = NLLLoss(k_logits, src_y[:, 1:].contiguous().view(-1))
            bow_loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            if step % 50 == 0:
                print("Epoch [%.1d/%.1d] Step [%.4d/%.4d]: bow_loss=%.4f" % (epoch + 1, args.pre_epoch,
                                                                             step, len(train_loader),
                                                                             bow_loss.item()))


def train(model, optimizer, train_loader, args):
    encoder, Kencoder, manager, decoder = [*model]
    encoder.train(), Kencoder.train(), manager.train(), decoder.train()
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters()) + list(decoder.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    KLDLoss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(args.n_epoch):
        for step, (src_X, src_y, src_K, tgt_y) in enumerate(train_loader):
            src_X = src_X.cuda()
            src_y = src_y.cuda()
            src_K = src_K.cuda()
            tgt_y = tgt_y.cuda()

            optimizer.zero_grad()
            encoder_outputs, hidden = encoder(src_X)
            x = hidden
			
            n_batch = src_X.size(0)
            max_len = tgt_y.size(1)
            n_vocab = decoder.n_vocab
			
            CLS_tokens = torch.LongTensor([params.CLS] * n_batch).unsqueeze(1).cuda()
            SEP_tokens = torch.LongTensor([params.SEP] * n_batch).unsqueeze(1).cuda()
            src_y_ = torch.cat((CLS_tokens, src_y[:, 1:], SEP_tokens), dim=-1)
			
            y = Kencoder(src_y_)
            K = Kencoder(src_K)
            prior, posterior, k_i, k_logits = manager(x, y, K)
            kldiv_loss = KLDLoss(prior, posterior.detach())

            seq_len = src_y.size(1) - 1
            k_logits = k_logits.repeat(seq_len, 1, 1).transpose(0, 1).contiguous().view(-1, n_vocab)
            bow_loss = NLLLoss(k_logits, src_y[:, 1:].contiguous().view(-1))


            outputs = torch.zeros(max_len, n_batch, n_vocab).cuda()
            output = src_y[:, 0]  # [n_batch]
            hidden = hidden.unsqueeze(0)
            for t in range(max_len):
                output, hidden, attn_weights = decoder(output, k_i, hidden, encoder_outputs)
                outputs[t] = output
                is_teacher = random.random() < args.tfr  # teacher forcing ratio
                top1 = output.data.max(1)[1]
                output = tgt_y[:, t] if is_teacher else top1

            outputs = outputs.transpose(0, 1).contiguous()
            nll_loss = NLLLoss(outputs.view(-1, n_vocab),
                               tgt_y.contiguous().view(-1))

            loss = kldiv_loss + nll_loss + bow_loss
            loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()

            if step % 50 == 0:
                print("Epoch [%.2d/%.2d] Step [%.4d/%.4d]: total_loss=%.4f kldiv_loss=%.4f bow_loss=%.4f nll_loss=%.4f"
                      % (epoch + 1, args.n_epoch,
                         step, len(train_loader),
                         loss.item(), kldiv_loss.item(),
                         bow_loss.item(), nll_loss.item()))


def main():
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
    temperature = params.temperature
    train_path = params.train_path
    assert torch.cuda.is_available()

    print("loading_data...")

    train_X, train_y, train_K = load_data(train_path)
    train_loader = get_data_loader(train_X, train_y, train_K, n_batch)
    print("successfully loaded")

    encoder = Encoder().cuda()
    Kencoder = KnowledgeEncoder(n_hidden).cuda()
    manager = Manager(n_hidden, n_vocab, temperature).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden).cuda()

    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        Kencoder = torch.nn.DataParallel(Kencoder)
        manager = torch.nn.DataParallel(manager)
        decoder = torch.nn.DataParallel(decoder)

    if args.restore:
        encoder = init_model(encoder, restore=params.encoder_restore)
        Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
        manager = init_model(manager, restore=params.manager_restore)
        decoder = init_model(decoder, restore=params.decoder_restore)

    model = [encoder, Kencoder, manager, decoder]
    parameters = list(encoder.parameters()) + list(Kencoder.parameters()) + \
                 list(manager.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    # pre_train knowledge manager
    print("start pre-training")
    pre_train(model, optimizer, train_loader, args)
    print("start training")
    train(model, optimizer, train_loader, args)

    # save final model
    save_model(encoder, "PostKS-encoder.pt")
    save_model(Kencoder, "PostKS-Kencoder.pt")
    save_model(manager, "PostKS-manager.pt")
    save_model(decoder, "PostKS-decoder.pt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

