''' Handling the data io '''
import argparse
import torch
import math
import random
import numpy as np
import transformer.Constants as Constants

def preprocess_for_pretrain(lines, mask_rate, in_statement_pred):
    lines = lines.strip().split("<nl> ")
    
    addition_idx = []
    deletion_idx = []
    source_list = []
    target_list = []
    before_commit = []
    after_commit = []

    for idx, line in enumerate(lines):
        if line[0] == "+":
            after_commit.append(line[1:])
            addition_idx.append(idx)
        elif line[0] == "-":
            before_commit.append(line[1:])
            deletion_idx.append(idx)
        else:
            after_commit.append(line)
            before_commit.append(line)
    
    # handling commits with implicit binary file changes
    if before_commit == after_commit:
        before_commit_len = [len(src.split()) for src in before_commit]
        maxlen = max(before_commit_len)
        maxlen_idx = np.asarray(before_commit_len).argmax()

        mask_len = math.floor(mask_rate*maxlen)
        mask_start = random.randint(0, maxlen-mask_len)
        mask_end = mask_start + mask_len
        line_to_mask = before_commit[maxlen_idx].split()

        for i in range(mask_start, mask_end):
            line_to_mask[i] = Constants.MSK_WORD
        before_commit[maxlen_idx] = " ".join(line_to_mask)

    source_list.append("<nl> ".join(before_commit))
    target_list.append("<nl> ".join(after_commit))

    if in_statement_pred:
        # for idx_to_mask in range(len(lines)):
        for idx_to_mask in addition_idx + deletion_idx:
            token_mask_source = []
            token_mask_target = []
            
            line_to_mask = lines[idx_to_mask].split()
            mask_len = math.floor(mask_rate*len(line_to_mask))
            token_to_mask_idx = random.randint(0, len(line_to_mask) - 1)
            for idx in range(len(lines)):
                if idx == idx_to_mask:
                    # token_mask_source.append( " ".join([Constants.MSK_WORD if idx in range(token_to_mask_idx, token_to_mask_idx + mask_len) else token for idx, token in enumerate(line_to_mask)]) )
                    token_mask_source.append( " ".join([Constants.MSK_WORD if idx == token_to_mask_idx else token for idx, token in enumerate(line_to_mask)]) )
                    token_mask_target.append(lines[idx])
                else:
                    token_mask_source.append( " ".join([Constants.PAD_WORD for token in lines[idx].split()]) )
                    token_mask_target.append( " ".join([Constants.PAD_WORD for token in lines[idx].split()]) )
            
            source_list.append(" <nl> ".join(token_mask_source))
            target_list.append(" <nl> ".join(token_mask_target))
    
    return source_list, target_list


def read_instances_from_file(inst_file, max_sent_len, keep_case, mask_rate, in_statement_pred):
    ''' Convert file into word seq lists and vocab '''

    source_insts = []
    target_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            source_list, target_list = preprocess_for_pretrain(sent, mask_rate, in_statement_pred)
            for source, target in zip(source_list, target_list):
                source_words = source.split()
                target_words = target.split()

                if len(source_words) > max_sent_len or len(target_words) > max_sent_len:
                    trimmed_sent_count += 1
                source_inst = source_words[:max_sent_len]
                target_inst = target_words[:max_sent_len]

                if source_inst:
                    source_insts += [[Constants.BOS_WORD] + source_inst + [Constants.EOS_WORD]]
                else:
                    source_insts += [None]
                if target_inst:
                    target_insts += [[Constants.BOS_WORD] + target_inst + [Constants.EOS_WORD]]
                else:
                    target_insts += [None]


    print('[Info] Get {} instances from {}'.format(len(source_inst), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return source_insts, target_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.MSK_WORD: Constants.MSK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-vocab', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-mask_rate', type=float, default=0.3)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-in_statement_pred', type=bool, default=False)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts, train_tgt_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case, opt.mask_rate, opt.in_statement_pred)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts, valid_tgt_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case, opt.mask_rate, opt.in_statement_pred)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    predefined_data = torch.load(opt.vocab)
    assert 'dict' in predefined_data

    print('[Info] Pre-defined vocabulary found.')
    src_word2idx = predefined_data['dict']['src']
    tgt_word2idx = predefined_data['dict']['tgt']


    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
