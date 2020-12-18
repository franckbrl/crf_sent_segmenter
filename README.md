# crf_sent_segmenter

A CRF model for text segmentation into sentences.

## Requirements

This implementation is in python3. It is based on [Scikit-learn](https://scikit-learn.org/stable/), [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) and requires [mosestokenizer](https://pypi.org/project/mosestokenizer/) for text preprocessing.

```
pip3 install -U scikit-learn
pip3 install sklearn-crfsuite
pip3 install mosestokenizer
```

## Training

For training, run `train_sentence_segmenter.py`. The minimal input is the training data. The minimal ouput is the model trained on this data.

The training data (and optional test data) should be in [CONLL-U format](https://universaldependencies.org/format.html). Sentences from the data are concatenated in paragraphs. Paragraph lengths are set randomly (see `--max-plen` argument). Thus the model is optimized to retrieve the sentence boundaries in such paragraphs.

For more details and options, run `train_sentence_segmenter.py --help`.

## Inference

For inference, run `run_sentence_segmenter.py`. It takes as input a text file and returns one split sentence per line.

For more details and options, run `run_sentence_segmenter.py --help`.

## Quick example

### Just run inference

```
echo "Faut arrêter ces conneries de nord et de sud. Une fois pour toutes, le nord, suivant comment on est tourné, ça change tout." | python3 run_sentence_segmenter.py --model model_fr/sseg_model.pkl
```
The model provided in `model_fr/sseg_model.pkl` was trained in the same way as shown below.

### Train a model

* Install required modules
* Download training and test data: [French Sequoia corpus](http://deep-sequoia.inria.fr/) from Universal Dependencies project. In the root of this repository, run:
```
mkdir data
wget -P data/ https://raw.githubusercontent.com/UniversalDependencies/UD_French-Sequoia/master/fr_sequoia-ud-train.conllu data/
wget -P data/ https://raw.githubusercontent.com/UniversalDependencies/UD_French-Sequoia/master/fr_sequoia-ud-test.conllu data/
```
* Run training with default hyper-parameters (this shouldn't take more than a few minutes on an average laptop):
```
python3 train_sentence_segmenter.py --train data/fr_sequoia-ud-train.conllu --test data/fr_sequoia-ud-test.conllu
```
* Run inference (provided the model is in the current directory):
```
echo "Faut arrêter ces conneries de nord et de sud. Une fois pour toutes, le nord, suivant comment on est tourné, ça change tout." | python3 run_sentence_segmenter.py
```
