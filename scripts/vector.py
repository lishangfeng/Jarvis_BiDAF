#!/usr/bin/enc python3
# vector script of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

from collections import Counter
import torch

def vectorize(ex, model, single_answer=False):
    """
    Torchify a single example.
    """
    args = model.args
    word_dict = model.word_dict
    char_dict = model.char_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    # Index characters
    document_char = None
    question_char = None
    if args.use_char_emb:
        document_char = torchify_char(ex['document'], char_dict)
        question_char = torchify_char(ex['question'], char_dict)

    # Create extra features vectors
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        feature = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, ex['id']

    # or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, features, question, start, end, document_char, question_char, ex['id']

def torchify_char(text, char_dict):
    """
    Torchify the character.
    """
    chars_text = [list(w) for w in text]
    length = len(text)
    chars_text_idx = torch.zeros(length, 16, dtype = torch.long)
    # chars_text_mask = torch.ByteTensor(length, 16).fill_(1)
    for i, chars_word in enumerate(chars_text):
        chars_word_idx = torch.LongTensor([char_dict[c] for c in chars_word])
        chars_text_idx[i][:len(chars_word_idx)].copy_(chars_word_idx[:16])
        # chars_text_mask[i][:len(chars_word_idx)].fill_(0)
    return chars_text_idx #, chars_text_mask

def batchify(batch):
    """
    Gather a batch of individual examples into one batch.
    """
    NUM_INPUTS = 5
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]

    context_chars_idx = [ex[5] for ex in batch]
    query_chars_idx = [ex[6] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    context = torch.LongTensor(len(docs), max_length).zero_()
    context_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    
    if features[0] is None:
        context_feature = None
    else:
        context_feature = torch.zeros(len(docs), max_length, features[0].size(1))
    
    if context_chars_idx[0] is None:
        context_chars = None
    else:
        context_chars = torch.zeros(len(docs), max_length, 16, dtype = torch.long)
    
    for i, d in enumerate(docs):
        context[i,:d.size(0)].copy_(d)
        context_mask[i, :d.size(0)].fill_(0)
        if context_feature is not None:
            context_feature[i, :d.size(0)].copy_(features[i])
        if context_chars is not None:
            context_chars[i, :d.size(0)].copy_(context_chars_idx[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    query = torch.LongTensor(len(questions), max_length).zero_()
    query_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    if query_chars_idx[0] is None:
        query_chars = None
    else:
        query_chars = torch.zeros(len(questions), max_length, 16,  dtype = torch.long)
    for i, q in enumerate(questions):
        query[i, :q.size(0)].copy_(q)
        query_mask[i, :q.size(0)].fill_(0)
        if query_chars is not None:
            query_chars[i, :q.size(0)].copy_(query_chars_idx[i])

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return context, context_feature, context_mask, query, query_mask, context_chars, query_chars, ids
        # return context, context_feature, query, ids
    
    elif len(batch[0]) == NUM_INPUTS + NUM_TARGETS + NUM_EXTRA:
        # Otherwise add targets
        if torch.is_tensor(batch[0][3]):
            target_s = torch.cat([ex[3] for ex in batch])
            target_e = torch.cat([ex[4] for ex in batch])
        else:
            target_s = [ex[3] for ex in batch]
            target_e = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')
    return context, context_feature, context_mask, query, query_mask, context_chars, query_chars, target_s, target_e, ids



                