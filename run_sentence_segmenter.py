#!/usr/bin/python3

import sys
import pickle
import argparse
import html

from features import paragraph2features

import mosestokenizer as mt


class preproc():
    def __init__(self, lang):
        self.normalize = mt.MosesPunctuationNormalizer(lang)
        self.tokenize = mt.MosesTokenizer(lang)

    def run_normalization(self, text):
        norm_text = self.normalize(text)
        norm_text = norm_text.replace("’", "'")
        self.check_proc(text, norm_text)
        return norm_text

    def run_tokenization(self, text):
        tokens = self.tokenize(text)
        tokens = html.unescape(" ".join(tokens))
        tokens = tokens.replace(" @-@ ", "-")
        self.check_proc(text, tokens)
        return tokens.split()

    @staticmethod
    def check_proc(before, after):
        """
        Ensure pre-processing step did not add or remove
        any character (other than space).
        """
        len_before = len(before.replace(" ", ""))
        len_after = len(after.replace(" ", ""))
        assert len_before == len_after


def preprocess_input(text, pproc):
    """
    Get normalized and tokenized paragraphs
    """
    for parag in text:
        parag = parag.strip()
        if parag == "":
            continue
        pp_par = pproc.run_normalization(parag)
        pp_par = pproc.run_tokenization(parag)
        yield parag, pp_par


def split_sentences(parag, pp_par, labels):
    """
    Split initial raw sentences according to predicted labels.
    """
    sent = ""
    for tok, label in zip(pp_par, labels):
        sent += parag[:len(tok)]
        parag = parag[len(tok):]
        while parag.startswith(" "):
            sent += " "
            parag = parag[1:]
        if label == "1":
            yield sent
            sent = ""
    if sent:
        yield sent


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='inp', nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin, help="input data")
    parser.add_argument('-o', dest='out', nargs='?',
                        type=argparse.FileType('w'),
                        default=sys.stdout, help="output data")
    parser.add_argument('--model', dest='model',
                        nargs='?', type=str,
                        default="sseg_model.pkl",
                        help="Name of model file (default: %(default)s).")
    parser.add_argument('--lang', dest='lang',
                        nargs='?', type=str,
                        default="fr",
                        help="Preprocessing language (default: %(default)s).")
    return parser.parse_args()


def main(args):
    # pre-processing steps
    pproc = preproc(args.lang)
    # load model
    with open(args.model, "rb") as ss:
        crf = pickle.load(ss)
    # predict sentence boundaries
    for parag, pp_par in preprocess_input(args.inp, pproc):
        feat = [paragraph2features(pp_par)]
        labels = crf.predict(feat)[0]
        sents = split_sentences(parag, pp_par, labels)
        for sent in sents:
            args.out.write(sent + "\n")


main(arguments())
