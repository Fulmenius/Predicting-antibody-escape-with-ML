# Predicting-antibody-escape-with-ML
Predicting antibody escape and ACE2 binding of COVID-19 spike protein based on the amino-acid sequence of RBD

## Introduction
At the early stages of the COVID-19 pandemic, monoclonal antibodies were probably the only effective non-symptomatic treatment of the disease.[] Later on, the performance of vaccines against the coronavirus was estimated, among other things, from the production of antibodies against the virus after vaccination, as they were considered to be one of the key mechanisms by which vaccination prevents and/or lightens the infection.[] However, COVID-19 mutates fairly quickly, and some of the new strains develop the ability to escape antibodies effective against the original Wuhan strain, both obtained the "natural" way and with the help of the vaccines.[] This process may render some of the current protocols for treatment and prevention of the disease outdated, and hence poses a serious threat to the world health security. Hence, it would be extremely useful to know in advance, without having to perform the tests __in vitro__, what possible mutations of the coronavirus may help it escape the immunity created by vaccines and infections with older strains.

This is the problem that Taft et al. addressed in their paper []. Using both classical Machine Learning and Deep Learning models, and a vast database of mutation effects obtained via yeast surface display[], they were able to predict the effect of both singular and combinatorial mutations on the ability of the four antibodies used in clinical practice to bind to a mutated receptor-binding domain (RBD) of SARS-CoV 2 S-protein, with precision exceeding 90% and reaching 100% in some problems. 

This project's aim is twofold. At one hand, it is a simple reproduction of results of Taft et al. At the other, there is some hope of improving these results by using more advanced Machine Learning techniques. Taft et al. used one-hot encoding as a vector representation for the amino-acid sequence of the mutated region of RBD. This representation does not represent well the chemical and biological structure of the space of amino acid sequences. A better way to do that is provided by the use of autoencoders - userupervisedly learned vector representations of data, that have been used with great success in the field of Natural Language Processing. Such models can learn grammar of natural languages and represent it using the spatial structure of the vector representation. Elnaggar et al.[PT], among others, used state of the art autoencoders and auto-regressive models, such as Transformers, to create vector representations for protein sequences that reflect the chemical and biological structure of the "protein space", i.e., the "grammar" of the "language of life". Using these pretrained encodings as inputs for other models, a technique called Transfer Learning, they managed to significantly improve the performance of these models. The ultimate goal of our project is to do the same in the context of work of Taft et al.

## Formulation of the problem and the data

What we want to solve is essentialy two problems of binary classification: prediction 

## The model

## Results

## Discussion

