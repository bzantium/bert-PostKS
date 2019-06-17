import torch
import params
from utils import init_model, knowledgeToIndex
from pytorch_pretrained_bert import BertTokenizer
from model import Encoder, KnowledgeEncoder, Decoder, Manager

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def main():
    max_len = 50
    n_vocab = params.n_vocab
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    temperature = params.temperature
    assert torch.cuda.is_available()

    print("loading model...")
    encoder = Encoder().cuda()
    Kencoder = KnowledgeEncoder(n_hidden).cuda()
    manager = Manager(n_hidden, n_vocab, temperature).cuda()
    decoder = Decoder(n_vocab, n_embed, n_hidden).cuda()

    encoder = init_model(encoder, restore=params.encoder_restore)
    Kencoder = init_model(Kencoder, restore=params.Kencoder_restore)
    manager = init_model(manager, restore=params.manager_restore)
    decoder = init_model(decoder, restore=params.decoder_restore)
    print("successfully loaded!\n")

    utterance = ""
    while True:
        if utterance == "exit":
            break
        k1 = input("Type first Knowledge: ").lower()
        k2 = input("Type second Knowledge: ").lower()
        k3 = input("Type third Knowledge: ").lower()

        K = [k1, k2, k3]
        K = knowledgeToIndex(K)
        K = Kencoder(K)
        print()

        while True:
            utterance = input("you: ").lower()
            if utterance == "change knowledge" or utterance == "exit":
                print()
                break

            tokens = tokenizer.tokenize(utterance)
            seqs = tokenizer.convert_tokens_to_ids(tokens)
            X = torch.LongTensor(seqs).unsqueeze(0).cuda()  # X: [1, x_seq_len]

            encoder_outputs, hidden = encoder(X)
            k_i = manager(hidden, None, K)
            outputs = torch.zeros(max_len, 1, n_vocab).cuda()  # outputs: [max_len, 1, n_vocab]
            output = torch.LongTensor([params.SOS]).cuda()

            for t in range(max_len):
                output, hidden, attn_weights = decoder(output, k_i, hidden, encoder_outputs)
                outputs[t] = output
                output = output.data.max(1)[1]

            outputs = outputs.max(2)[1]

            tokens = []
            for idx in outputs:
                if idx == params.EOS:
                    break
                tokens += tokenizer.convert_ids_to_tokens([idx])[0]

            text = ' '.join([x for x in tokens])
            answer = text.replace(' ##', '')
            print("bot:", answer, "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
