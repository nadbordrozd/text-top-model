Text Top Model is a tool for benchmarking text classification algorithms (especially different 
neural network architectures). This repository contains scripts for getting and preparing text 
classification datasets, implementation of several popular classification algorithms, a script 
for tuning hyperparameters and finally a script for running the benchmark and a jupyter notebook 
for plotting the results. 
 
### Gettting the data
To get all the necessary datasets, run:
```bash
bash get_data.sh
```
(you will need the `unzip` and `tar` tools for unpacking some of the datasets). The script will download several popular labeled text classification datasets as well as GloVe embeddings used by some of the algorithms.

Each dataset should be saved in a single file with one document per line. Each line should consist of the label, separated by a tab from tokens, which are separated by spaces. Like this:

```text
NEGATIVE the thing looks like a made for home video quickie
POSITIVE illuminating if overly talky documentary
POSITIVE a masterpiece four years in the making
```

Not all of the downloaded datasets are in this format. To transform them, run:

```bash
python prepare_polarity_dataset.py
python prepare_subjectivity_dataset.py
```

### Running the benchmark
There are currently 2 benchmark scripts - one for document and one for sentence classification. 
You can run them with:
```bash
cd ttm
python run_document_benchmark.py
python run_sentence_benchmark.py
```
The reason for having separate benchmarks is that the algorithms (and hyperparameters) that work well for sentence classification often give poor results (or take an unreasonable amount of time to train) for document classification.

Both benchmarking scripts can be easily expanded to add new models or datasets. To expand the 
benchmark, just pick a model and a dataset and the number of iterations and the `benchmark` 
function, like this:

```python
from ttm.benchmarks import benchmark
from ttm.keras_models.fchollet_cnn import FCholletCNN

model_class = FCholletCNN
model_params = {'dropout_rate': 0.5, 'embedding_dim': 37, 'units': 400, 'epochs': 30}
data_path = 'data/r8-all-terms.txt'
iterations = 10
list_of_scores, list_of_times = benchmark(model_class, data_path, model_params, iterations)
```

For this to work, the model must follow the scikit-learn API with `.fit(X, y)` and `.predict(X)` and `.get_params(recursive)` methods. All the texts are already tokenised (and perhaps stemmed or lemmatised), so the `X` in `.fit(X, y)` is a list of lists of token IDs, not actual strings.

Node that the `benchmark` function is cached to disk with joblib, so if you add a model or dataset to the script and execute it, only the new benchmarks will run. 

### Models
The models currently included in the benchmarks are sklearn's `SVM`, `BernoulliNaiveBayes`, `MultinomialNaiveBayes` and a bunch of neural network based models. The neural models live and `ttm/keras_models` (there are also pure Tensorflow implementations of most of them in `ttm/tflearn_models` courtesy of Javier Rodriguez Zaurin). 

The neural models included are: a multi-layer perceptron `MLP`, two different types of convolutional nets `FCholletCNN`, `YKimCNN`, a recurrent network LSTM/BLSTM `LSTM`, and a combination of a recurrent and convolutional net `BLSTM2DCNN`. See the classes in `ttm/keras_models` for links to papers. All the models are parametrised with the number of layers, number of units per layer, dropout rates etc. 
 
Finally, `ttm/stacking_classifier` has an implementation of stacking to combine multiple models.

### Results
A sample of results for document classification:
```
model            r8-all-terms.txt    r52-all-terms.txt    20ng-all-terms.txt    webkb-stemmed.txt
-------------  ------------------  -------------------  --------------------  -------------------
MLP 1x360                   0.966                0.935                 0.924                0.930
SVM tfidf bi                0.966                0.932                 0.920                0.911
SVM tfidf                   0.969                0.941                 0.912                0.906
MLP 2x180                   0.961                0.886                 0.914                0.927
MLP 3x512                   0.966                0.927                 0.875                0.915
CNN glove                   0.964                0.920                 0.840                0.892
SVM bi                      0.953                0.910                 0.816                0.879
SVM                         0.955                0.917                 0.802                0.868
MNB                         0.933                0.848                 0.877                0.841
CNN 37d                     0.931                0.854                 0.764                0.879
MNB bi                      0.919                0.817                 0.850                0.823
MNB tfidf tri               0.808                0.685                 0.866                0.762
MNB tfidf                   0.811                0.687                 0.843                0.779
MNB tfidf bi                0.807                0.685                 0.855                0.763
BNB                         0.774                0.649                 0.705                0.741
BNB tfidf                   0.774                0.649                 0.705                0.741
```
and for sentence classification:
```
model               subjectivity_10k.txt    polarity.txt
----------------  ----------------------  --------------
Stacker LogReg                     0.935           0.807
Stacker XGB                        0.932           0.793
MNB 2-gr                           0.921           0.782
MNB tfidf 2-gr                     0.917           0.785
MNB tfidf 3-gr                     0.916           0.781
MNB tfidf                          0.919           0.777
MNB                                0.918           0.772
LSTM GloVe                         0.921           0.765
BLSTM Glove                        0.917           0.766
SVM tfidf 2-gr                     0.911           0.772
MLP 1x360                          0.910           0.769
MLP 2x180                          0.907           0.766
MLP 3x512                          0.907           0.761
SVM tfidf                          0.905           0.763
BLSTM2DCNN GloVe                   0.894           0.746
CNN GloVe                          0.901           0.734
SVM                                0.887           0.743
LSTM 12D                           0.891           0.734
CNN 45D                            0.893           0.682
LSTM 24D                           0.869           0.703
BLSTM2dCNN 15D                     0.867           0.656
```
For full results, see `ttm/document_results.csv` and `ttm/sentence_results.csv`. For plots see `ttm/evil_plotting.ipynb`.

**NOTE**: we have added a `ttm/tflearn_models` directory, using tensorflow's high level API `TFLearn`. At this stage, this is included *only* for exploratory purposes. However, the code there is relatively clean and usable, following the exact same structure than that of the `keras_model` directory.
