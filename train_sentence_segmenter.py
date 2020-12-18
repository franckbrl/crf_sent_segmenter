#!/usr/bin/python3

import pickle
import argparse
import random

from features import paragraph2features, paragraph2labels

import scipy.stats
import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import metrics


def prepare_data(text, max_plen=None):
    text = [p for p in generate_ud_data(text, max_plen)]
    x_text = [paragraph2features(p) for p in text]
    y_text = [paragraph2labels(p) for p in text]        
    return x_text, y_text

def generate_ud_data(ud_data, max_plen):
    """
    Generate paragraphs from UD data by concatenating
    sentences, keeping sentence boundaries as a label.
    """
    paragraph  = []
    labels     = []
    sent_added = 0
    paragraph_len = set_paragraph_len(max_plen)
    for token in ud_data:
        if token.startswith("#"):
            continue
        if token == "\n":
            sent_added += 1
            labels[-1] = "1"
            if sent_added == paragraph_len:
                yield [(p, l) for p, l in zip(paragraph, labels)]
                paragraph  = []
                labels     = []
                sent_added = 0
                paragraph_len = set_paragraph_len(max_plen)
                continue
        else:
            paragraph.append(token.split()[1])
            labels.append("0")
    if not max_plen:
        yield [(p, l) for p, l in zip(paragraph, labels)]

def set_paragraph_len(max_plen):
    """
    Paragraph length is set randomly and uniformly
    ranges from 1 to `max_plen`.
    If `max_plen` is not set, return infinity
    (recommended at test time).
    """
    if max_plen:
        return random.randrange(1, max_plen + 1)
    else:
        return float("inf")

def build_model(cv_folds, n_iter):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=n_iter,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }
    # Metric for cross-validation: macro avrg F1.
    # Label "1" is rare and difficult and thus
    # not under-represented, compared to easier "0".
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='macro', labels=["0", "1"])

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=cv_folds,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=n_iter,
                            scoring=f1_scorer)
    return rs

def run_test(test, model):
    x_test, y_test = prepare_data(test)
    y_pred = model.predict(x_test)
    print("Best model preformance on testset ({}):".format(test.name))
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=["0", "1"], digits=3
    ))

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', nargs='?', type=argparse.FileType('r'),
                        required=True, help="input training data.")
    parser.add_argument('--test', dest='test', nargs='?', type=argparse.FileType('r'),
                        help="test data")
    parser.add_argument('--max-plen', dest='max_plen', nargs='?', type=int,
                        default=10, help="Maximum length of paragraph in sentences (default: %(default)s).")
    parser.add_argument('--cv-folds', dest='cv_folds', nargs='?', type=int,
                        default=3, help="Number of folds for cross-validation (default: %(default)s).")
    parser.add_argument('--n-iter', dest='n_iter', nargs='?', type=int,
                        default=50, help="Number of iterations over training data (default: %(default)s).")
    parser.add_argument('--model-name', dest='model_name', nargs='?', type=str,
                        default="sseg_model.pkl",
                        help="Name of output file containing the model (default: %(default)s).")
    return parser.parse_args()

def main(args):
    x_train, y_train = prepare_data(args.train, args.max_plen)
    model = build_model(args.cv_folds, args.n_iter)
    # start training
    model.fit(x_train, y_train)
    print("Best hyper-parameters found:", model.best_params_)
    print("Best score: {:.2f}".format(model.best_score_ * 100))
    # output best model
    best_model = model.best_estimator_
    with open(args.model_name, "wb") as sseg:
        pickle.dump(best_model, sseg)
    # test best model
    if args.test:
        run_test(args.test, best_model)

        
main(arguments())










