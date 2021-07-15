# Unsupervised-Text-Classification

NLP Text classification
This Python module addresses a common problem of unsupervised text classification. In research projects I had to try different approaches so decided to aggregate all common options to faster narrow down on the one(s) working best for the problem at hand.
Classification flow is implemented via typical 4-step process:
1)	Vectorization -> 2) Feature Reduction -> 3) Outlier Removal -> 4) Final Clustering
For data in this example, I used datasets from 20 newsgroups.
Supported options for each of the above steps are:
1)	Vectorization:
a.	TFIDF (term-frequency - inverse document-frequency)
b.	Count

2)	Feature reduction: 
a.	PCA (Principal Component Analysis)
b.	NMF (Non-Negative Matrix Factorization)
c.	LDA (Latent Dirichlet Allocation)
d.	SVD (singular value decomposition)
e.	Multivariate Bernoulli model

3)	Outliner Remover (clustering of the entire set to identify outliner):
a.	BIRCH (Clustering Feature Tree)
b.	OPTICS
c.	DBSCAN (Density-Based Spatial Clustering of Applications)

4)	Final Clustering:
a.	K-means
b.	Gaussian Mixture
  
The module uses Silhouette, Calinski & Harabasz, and Davies & Bouldin scoring for evaluating accuracy of the classification(s)
