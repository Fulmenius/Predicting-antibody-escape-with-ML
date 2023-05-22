# Predicting-antibody-escape-with-ML
Predicting antibody escape and ACE2 binding of COVID-19 spike protein based on the amino-acid sequence of RBD

## Introduction
### Why do we need to predict antibody escape?
Monoclonal antibodies are one of the effective non-symptomatic treatments of the coronavirus disease.[CT23] The performance of vaccines against the coronavirus was estimated, among other things, from the production of antibodies against the virus after vaccination, as they were considered to be one of the key mechanisms by which vaccination prevents and/or lightens the infection.[V] However, COVID-19 mutates fairly quickly, and some of the new strains develop the ability to escape antibodies effective against the original Wuhan strain, both obtained the "natural" way and with the help of the vaccines.[T] This process may render some of the current protocols for treatment and prevention of the disease outdated, and hence poses a serious threat to the world health security. Hence, it would be extremely useful to know in advance, without having to perform the tests _in vitro_, what possible mutations of the coronavirus may help it escape the immunity created by vaccines and infections with older strains.

This is the problem that Taft et al. addressed in their paper [T]. Using both classical Machine Learning and Deep Learning models, and a vast database of mutation effects obtained via yeast surface display [Y], they were able to predict the effect of both singular and combinatorial mutations on the ability of the four antibodies used in clinical practice to bind to a mutated receptor-binding domain (RBD) of SARS-CoV 2 S-protein, with precision exceeding 90% and reaching 100% in some problems. 

### Aim of the project
This project's aim is twofold. At one hand, it is a simple reproduction of results of Taft et al. At the other, there is some hope of improving these results by using more advanced Machine Learning techniques. Taft et al. used one-hot encoding as a vector representation for the amino-acid sequence of the mutated region of RBD. This representation does not represent well the chemical and biological structure of the space of amino acid sequences. A better way to do that is provided by the use of autoencoders - userupervisedly learned vector representations of data, that have been used with great success in the field of Natural Language Processing. Such models can learn grammar of natural languages and represent it using the spatial structure of the vector representation. Elnaggar et al.[PT], among others, used state of the art autoencoders and auto-regressive models, such as Transformers, to create vector representations for protein sequences that reflect the chemical and biological structure of the "protein space", i.e., the "grammar" of the "language of life". Using these pretrained encodings as inputs for other models, a technique called Transfer Learning, they managed to significantly improve the performance of these models. The ultimate goal of our project is to do the same in the context of work of Taft et al.


## Formulation of the problem and description of the data
### Biological background
Antibodies are proteins produced by immune system to bind to specific molecules, normally those foreign to the organism. Each antibody is tailored to "recognize" a specific molecule, called the _antigene_ of the antibody. By binding their respective antigenes, antibodies mark them for destruction by other components of the immune system, and sometimes physically prevent the antigenes from performing their normal functions. If the antigene is large, like protein, the antibody usually recognizes only a certain part of the antigene, called _epitope_. In case of SARS CoV 2, one of the most effective choices of antigenes for the immune system to react to is its so-called spike protein (S-protein), which the coronavirus uses to bind to a specific human receptor protein embedded into the membrane of certain types of human cells - the ACE2 receptor. The part of the coronavirus S-protein which directly interacts with the ACE2 receptor is thus called the receptor-binding domain, or RBD for short. Binding to the RBD and physically blocking it, an antibody is able not only to mark the viral particle for destruction, but also prevent infection of human cells. Thus, antibodies which have the RBD as their epitope are used in clinical practice for coronavirus treatment. 

Each protein is a long chain of 20 elementary building blocks, (alpha-)amino acids (or a complex of such chains). 
Since binding of antibodies to their epitopes is highly specific, a random mutation in the amino-acid chain of the RBD can enable the coronavirus to escape such antibodies. However, a mutation can also hinder the ability of RBD to bind to the ACE2 receptor, making the virus unable to reproduce. Thus, a potentially dangerous mutation is such that enables the virus to escape antibodies, while preserving its ability to bind to ACE2 receptors. Our goal is to predict both based on the amino acid sequence of the mutated RBD. More specifically, we use one of subsequences of the RBD which are particularly important to ACE2 binding (called core regions of the receptor-binding motif), RBM-2, 24 amino acids long. 

### Composition of data and problem formulation
The data we are given with consists of 5 datasets, containing information on binding of ACE2 receptors and four kinds of antibodies used in clinical practice to the mutated RBD: LY16, LY555, REGN33 and REGN87, together with the corresponding mutated RBM-2 sequences.
The mutated sequeces of RBM-2 are represented as strings of capital letters of the latin alphabet 24 characters long. Binding or non-binding is represented with binary labels (1 for binding, 0 for non-binding). The datasets also contain edit distances from the mutated RBM-2s to the reference RBM-2 of the original Wuhan strain. There are other data fields, like "consensus count", which we are not interested in. 

Information on binding of mutated RBDs is obtained experimentally by Taft et al., via technique called yeast surface display - the detailed description of experimental data acquisition can be found in the original publication [T]. The datasets used in this work, also created by Taft et al., can be found at https://github.com/LSSI-ETH/Taft_Weber_2021/blob/main/Supp.%20Table%204%20Model_Sequences.zip. 

All five datasets were splitted by the authors of the original paper in test, train and validate, with all three splits being balanced by the number of binding and non-binding variants. The train dataset sizes vary from about 15,000 entries (LY555) to about 407,000 entries (ACE2). The train/test split size ratio is 9:1. 

The problem thus splits into five separate problems of binary classification of 24-character strings with an alphabet of 20 symbols. 


## The model
### Technical background: vector representations of texts
#### One-hot encoding
Classification of textual data demands a vector representation of text. The simplest way to represent a text in an alphabet of N symbols as a list of vectors is one-hot encoding: each symbol of the alphabet is represented as a unit basis vector in the standard orthonormal basis of an N-dimensional Euclidean vector space. Then a string of k characters can be represented as an $N \times k$ binary matrix, with exactly one 1 and N-1 zeros at each column. We can feed this matrix to the input of a neural network or another ML model. 

#### Autoencoders and autoregressive models
A more elaborate, but also more fruitful way to encode textual data is to use the grammar of the text, if one is present. In Natural Language Processing, where this is always the case, powerful methods were developed to learn the grammatical structure of the text automatically, and use it to create vector representations that respect that structure. The two most successful types of models used to solve this problem are autoencoders and autoregressive models.

Autoencoders map one-hot encoded text into a space of different (usually smaller) dimension using a neural network (ecnoder), and then use another neual network (decoder) to map the result back to the original dimension. The difference between the original and the final image is used to determine the loss function, which is used to train the encoder and the decoder. After the training process is done, the encoder with frozen weights can be used to obtain a "compressed" vector representation of a text (which is shown to contain refelect some of the grammatical structure of the text), from which the original text can be restored (with a certain precision) using decoder.

Autoregressive models solve the problem of predicting the next part of the text (token) based on its previus parts. At each step, information about the previous tokens is contained in a vector representation of a fixed size, called the hidden state, which is updated when each new token is presented to the model; the next hidden state is a result of action of a neural network on the new token and the previous hidden state, and the prediction of the model is the result of action of another neural network on the hidden state. The error of prediction is used to train the model. After the training process is done, a text can be given to the input of the model, and the last hidden state after the model reads the whole text can be used as a vector representation of the text. 

#### Vector representations of biological sequences
Both autoencoders and autoregressive models are shown to learn the structure of natural languages, both in the sense that grammatically and lexically close texts are mapped to spatially close vectors, and in the sense that autoencoders and autoregressive models have shown great success solving problems of text generation, text analysis and translation. This created a hope to learn the structure of the "language" of biological sequences in the same way. Indeed, Elnaggar et al.[E] have trained state-of-the-art language processing models on a vast dataset of billions of amino acid sequences (UniRef 50, UniRef 100 and BFD), and have shown that a) clusters of vector representations correspond to certain chemically or biologically meaningful features of proteins, and b) using this models to preprocess inputs for other biological models (including those that predict binding between proteins), one can significantly improve the quality of these models. 

Elnaggar et al. have trained several text encoders, and decided that ProtT5, based on the T5 text-to-text model by Raffel et al.[T5] works the best, so we decided to use it for our problem. The pre-trained model is available at https://github.com/agemagician/ProtTrans. T5 is a Transformer-type model with the hidden state of length 1024, which uses amino acids as tokens. It means the model maps each amino acid in a sequence to a 1024-dimensional vector. The dimension of the vector is much greater than the dimension of the one-hot representation, because each vector contains information both about the amino acid itself and its context, i.e., what other amino acids it is adjacent to. A representation of a whole text can be obtained either by averaging representations of individual amino acids or by stacking/concatenating them. Thus, the input of our model, if we use ProtT5, is either a 1024-dimensional vector, or a matrix of dimensions 24x1024.


### Description of the model
#### Input
In our work we have used two types of models: those that work directly with one-hot encoding, and those that use the 1024-dimensional vector representation obtained through ProtT5. Thus, the input is either a matrix of dimensions 20x24, a 1024-dimensional vector, or a matrix of dimensions 24x1024. 

#### One-hot-encoding-based models
##### CNN
We've used CNN with 2 convolutional layers and 2 fully connected layers, with BatchNorm and MaxPooling applied between the convolutional layers. The network's architecture can be seen at the illuststation below. The model (not pre-trained) can be found in the /models folder.
![Figure 1: ConvNet architecture](/models/convnet_architecture.png)

##### XGBoost
We've used XGBoost model with the following parameters:
        objective="binary:logistic", 
        random_state=42,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=9,
        gamma=1,
        reg_alpha=0.005

#### ProtT5-based models
##### CNN
The network's architecture can be seen at the illuststation below.
The model (not pre-trained) can be found in the /models folder.
![Figure 2: ConvNet1D architecture](/models/convnet1d_architecture.png)

##### XGBoost
Parameters of XGBoost were the same as in the first example.

## Results
Contrary to our expectations, the vector representation not only didn't increase the average ROC/AUC score of the model, but decreased it, not much, but statistically significantly, for all pairs as per Welch's $t$-test. However, it also reduced the variance in prediction quality dramatically, making the model much more reliable. Comparison of results for $5$, $10$ and $20$ epochs indicates, that the embedding model overfitting is not likely to be the case of the prediction quality decrease in the model with embedding, since the difference in average quality of prediction between one-hot-using and embedding-using CNN does not change significantly. However underfitting might theoretically be the case for the embedding using model.

For some reason, the XGBoost model we used was completely unable to learn the pattern of binding. 

![Figure 3](/results/20_epochs_final_plot.png)
![Figure 4](/results/10_epochs_final_plot.png)
![Figure 5](/results/5_epochs_final_plot.png)


## Discussion

## Literature
[T] Predictive profiling of SARS-CoV-2 variants by deep mutational learning
Joseph M. Taft, Cédric R. Weber, Beichen Gao, Roy A. Ehling, Jiami Han, Lester Frei, Sean W. Metcalfe, Alexander Yermanos, William Kelton, Sai T. Reddy
bioRxiv 2021.12.07.471580; doi: https://doi.org/10.1101/2021.12.07.471580 

[E] Elnaggar A, Heinzinger M, Dallago C, Rehawi G, Wang Y, Jones L, Gibbs T, Feher T, Angerer C, Steinegger M, Bhowmik D, Rost B. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell. 2022 Oct;44(10):7112-7127. doi: 10.1109/TPAMI.2021.3095381. Epub 2022 Sep 14. PMID: 34232869.

[Y] Boder, Eric T.; Wittrup, K. Dane (1997). "Yeast surface display for screening combinatorial polypeptide libraries". Nature Biotechnology. 15 (6): 553–557. doi:10.1038/nbt0697-553.

[CT23] Murakami, N., Hayden, R., Hills, T. et al. Therapeutic advances in COVID-19. Nat Rev Nephrol 19, 38–52 (2023). https://doi.org/10.1038/s41581-022-00642-4

[V] Efficacy of SARS-CoV-2 vaccines and the dose–response relationship with three major antibodies: a systematic review and meta-analysis of randomised controlled trials, Yang et al., The Lancet, 2023
