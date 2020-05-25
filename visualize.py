''' Translate input text with trained translator.model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

from dataset import collate_fn, TranslationDataset
from transformer.Translator import Translator
from preprocess import read_instances_from_file, convert_instance_to_idx_seq
import transformer.Constants as Constants
from utils.postprocess import del_repeat

def read_instances(inst, max_sent_len, keep_case):
    ''' Convert string into word seq lists and vocab '''

    word_insts = []
    if not keep_case:
        inst = inst.lower()
    words = inst.split()
    word_inst = words[:max_sent_len]
    word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
    
    return word_insts

def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    ax=ax, cbar=True)

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-vocab', required=True,
                        help='Path to vocabulary file')
    parser.add_argument('-output',
                        help="""Path to output the predictions""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    src_line = "Binary files a / build / linux / jre . tgz and b / build / linux / jre . tgz differ <nl>"

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_src_word_insts = read_instances(
        src_line,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=1,
        collate_fn=collate_fn)

    translator = Translator(opt)


    for batch in tqdm(test_loader, mininterval=1, desc='  - (Test)', leave=False):
        all_hyp, all_scores = translator.translate_batch(*batch)
        for idx_seqs in all_hyp:
            for idx_seq in idx_seqs:
                pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq[:-1]])
            print(pred_line)
    
    sent = src_line.split()
    tgt_sent = pred_line.split()
    
    for layer in range(0, 2):
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        print("Encoder Layer", layer+1)
        for h in range(4):
            print(translator.model.encoder.layer_stack[layer].slf_attn.attn.data.cpu().size())
            draw(translator.model.encoder.layer_stack[layer].slf_attn.attn[h, :, :].data.cpu(), 
                sent, sent if h ==0 else [], ax=axs[h])
        plt.savefig(opt.output+"Encoder Layer %d.png" % layer)
        
    for layer in range(0, 2):
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        print("Decoder Self Layer", layer+1)
        for h in range(4):
            print(translator.model.decoder.layer_stack[layer].slf_attn.attn.data.cpu().size())
            draw(translator.model.decoder.layer_stack[layer].slf_attn.attn[:,:, h].data[:len(tgt_sent), :len(tgt_sent)].cpu(), 
                tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
        plt.savefig(opt.output+"Decoder Self Layer %d.png" % layer)

        print("Decoder Src Layer", layer+1)
        fig, axs = plt.subplots(1,4, figsize=(20, 10))
        for h in range(4):
            draw(translator.model.decoder.layer_stack[layer].slf_attn.attn[:,:, h].data[:len(sent), :len(tgt_sent)].cpu(), 
                tgt_sent, sent if h ==0 else [], ax=axs[h])
        plt.savefig(opt.output+"Decoder Src Layer %d.png" % layer)
                    
    print('[Info] Finished.')

if __name__ == "__main__":
    main()
