# chessAI

The preprocessed FEN data is contained in ``data_processed.npy.gz`` and ``results_processed.npy.gz``, containing the flattened $8\times8\times12$ one-hot encoding of the tensors and the labels, respectively. The notebook ``data.ipynb`` processes the data.

``dropout.py`` contains the code to run to obtain Figures __ and __ from the report, which give the training and test accuracies for the shallow and deep nets with different dropout rates.

``augmentations.py`` contains the functions that transform the dataset to use context and worldview learning, and ``data_augmentations.ipynb`` contains the code to run to obtain Figures _ and _ from the report. 
