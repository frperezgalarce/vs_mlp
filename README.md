# Informative regularization for a multi-layer perceptron RR Lyrae classifier under data shif

In recent decades, machine learning has provided valuable models and algorithms for processing
and extracting knowledge from time-series surveys. Different classifiers have been proposed and
performed to an excellent standard. Nevertheless, few papers have tackled the data shift problem
in labeled training sets, which occurs when there is a mismatch between the data distribution in the
training set and the testing set. This drawback can damage the prediction performance in unseen
data. Consequently, we propose a scalable and easily adaptable approach based on an informative
regularization and an ad-hoc training procedure to mitigate the shift problem during the training
of a multi-layer perceptron for RR Lyrae classification. We collect ranges for characteristic features
to construct a symbolic representation of prior knowledge, which was used to model the informative
regularizer component. Simultaneously, we design a two-step back-propagation algorithm to integrate
this knowledge into the neural network, whereby one step is applied in each epoch to minimize
classification error, while another is applied to ensure regularization. Our algorithm defines a subset of
parameters (a mask) for each loss function. This approach handles the forgetting effect, which stems
from a trade-off between these loss functions (learning from data versus learning expert knowledge)
during training. Experiments were conducted using recently proposed shifted benchmark sets for
RR Lyrae stars, outperforming baseline models by up to 3% through a more reliable classifier. Our
method provides a new path to incorporate knowledge from characteristic features into artificial neural
networks to manage the underlying data shift problem

## Pérez-Galarce, F., et al. "Informative regularization for a multi-layer perceptron RR Lyrae classifier under data shift." Astronomy and Computing (2023): 100694.