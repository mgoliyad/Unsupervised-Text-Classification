import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import  MiniBatchKMeans, DBSCAN, Birch, OPTICS, cluster_optics_dbscan
from sklearn.neural_network import BernoulliRBM
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import v_measure_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD, PCA
from sklearn.metrics import  silhouette_score, make_scorer
from sklearn.preprocessing import Normalizer
from preprocessing import get_train_data

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Unsupervised_Classifier():
 
    def __init__(self, data_dir, config, printout):
        self.vocab_size = config["text_processing"]["max_vocab_size"]
        self.min_text_length= config["text_processing"]["min_text_length"]
        self.min_vocab_size = config["text_processing"]["min_vocab_size"]
        self.min_portion_samples = config["unsupevised"]["min_portion_samples_to_keep"]
        self.min_frequency = config["unsupevised"]["min_frequency"]
        self.max_frequency = config["unsupevised"]["max_frequency"]
        self.printout = printout
        self.data_dir = data_dir
        self.min_k = config["unsupevised"]["min_k"]
        self.max_k = config["unsupevised"]["max_k"] + 1
        self.min_feature = config["text_processing"]["min_vocab_size"]
        self.clustering_model = config["unsupevised"]["clustering"]
        self.dim_reduction_model = config["unsupevised"]["dim_reduction_model"]
        self.vec_model = config["unsupevised"]["vec_model"]
        self.outliner_removal_model = config["unsupevised"]["outliner_removal_model"]
    

    def plot_top_words(self, components, model_name, feature_names, n_top_words=10):
        '''Plotting of top n_top_words words for first 10 clusters'''
        title = 'Topics in %s model'%model_name
     
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
     
        for topic_idx, topic in enumerate(components):
            if topic_idx >= len(axes) :
                break
     
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
     
            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Topic {topic_idx +1}',
                          fontdict={'fontsize': 10})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=10)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=20)
     
        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show() 
     
    def silhouette_scorer(self, X, y_pred):
        '''Scoring for unlabeled data''' 
        if len(set(y_pred)) == 1:
            return -1
        else:
            return silhouette_score(X, y_pred)

    def calinski_harabasz_scorer(self, X, y_pred):
        '''Scoring for unlabeled data''' 
        if len(set(y_pred)) == 1:
            return -1
        else:
            return calinski_harabasz_score(X, y_pred)

    def davies_bouldin_scorer(self, X, y_pred):
        '''Scoring for unlabeled data''' 
        if len(set(y_pred)) == 1:
            return -1
        else:
            return davies_bouldin_score(X, y_pred)
        
    def vectorizing(self, documents, min_df, max_df, prev_n_features, ):
        '''Picking right parameters to do vectorization with new number of features'''
        
        vectorizer_name = self.vec_model
    
        if vectorizer_name == 'TFIDF':
            vectorizer  = TfidfVectorizer(max_features=self.vocab_size, strip_accents = 'ascii', decode_error='ignore', \
                                 max_df=max_df, min_df=min_df, sublinear_tf=True, use_idf=True, stop_words='english')
        else:
            vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=self.vocab_size, stop_words='english')
    
        try:
            vectors = vectorizer.fit_transform(documents)
        except:
            '''Some values of min_df and max_df can eliminate all voctors so we just skip this scenario'''
            if self.printout == True:
                print('Unable to fit vectorizer = {} for max_df = {}, min_df = {}'.format(vectorizer_name, max_df, min_df))
            return [], [], 0
       
        X = vectors.toarray()         
        feature_names = vectorizer.get_feature_names()
        n_feature = len(feature_names)
        '''Our goal to get different vectorization each time to get more variety of training data.
        So, we continue only with new set of feature.'''
        
        if (n_feature in prev_n_features):
           return [], [], 0
        if self.printout == True:
            print('Vectorization completed with {} features, shape: {}'.format(n_feature, X.shape))
        return feature_names, X, n_feature
        
    
    
    def get_n_feature(self, X):
        '''Grid search for PCA model to get number of features with 90% of 
        explaine variance to use for feature_analysis function'''
        
        pca = PCA()
        param_grid = {'whiten': [True, False], 'svd_solver': ['auto', 'full', 'arpack', 'randomized']}
        grid_search = GridSearchCV(pca, param_grid=param_grid, n_jobs=-1)
       
        try:
            grid_search.fit(X)
        except:
            return 0
       
        best_estimator = grid_search.best_estimator_
        cumsum = np.cumsum(best_estimator.explained_variance_ratio_)
        return np.argmax(cumsum >=0.9 ) + 1
    
               
    def outliner_detection(self, model_name, X, y):
        ''' To reduce number of samples by removing of some outliners  '''
        
        min_num_samples = int(self.min_portion_samples * len(y))
       # print(min_num_samples)
        best_score = -1

        if 'birch' in model_name:
            '''GridSearch w/o specified number of clusters to let machine pick the best number. 
            We keep 90% of data closest to the center of each cluster.'''
            thresholds = [0.9, 0.8, 0.7]
            model = Birch(n_clusters=None)
            all_data = pd.DataFrame()
            param_grid = {'threshold': thresholds, 'n_clusters': [None]}
            scorer = make_scorer(self.silhouette_scorer)
            grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scorer, n_jobs=-1)
           
            grid_search.fit(X, X)
            best_estimator = grid_search.best_estimator_
            best_pred = best_estimator.predict(X)
            best_score = grid_search.best_score_
            
            if best_score == -1:
                return X, y
            
            n_clusters = len(set(best_pred))
              
            all_data = pd.DataFrame()
            all_data['label'] = best_pred
            all_data['y'] = y
            best_X = best_estimator.transform(X)
            all_data['dist'] = best_X.min(axis=1)
           
            mask = np.zeros(len(X), dtype=bool)

            for n in range(n_clusters):
                n_data = all_data.loc[all_data['label']==n].sort_values(by=['dist'])
            
                for i, f in n_data[:int(len(n_data)*0.9)].iterrows():
                    mask[i] = True
            best_X_no_outliner = np.array(X)[mask]
            best_y_no_outliner = np.array(y)[mask]
     
        elif 'optics' in model_name:
            '''Using different value of eps (maximum distance between two samples for one to be 
            considered as in the neighborhood) to pick the best model and then filtering all 
            data which was not assigned in any neighborhood. If too less samples were selected 
            then we skip this step and return origical data.'''
            
            min_samples = [min_num_samples, int(min_num_samples*0.95), int(min_num_samples*0.9)]
            param_grid = {'min_samples': min_samples}
            model = OPTICS(xi=.05, n_jobs=-1)
            for m in min_samples:
                
                model.set_params(min_samples=m)
                y_pred = model.fit_predict(X)
                score = self.silhouette_scorer(X, y_pred)
                if score > best_score:
                    best_score = score
                    labels = y_pred
                    
    
            labels_050 = cluster_optics_dbscan(reachability=model.reachability_,
                                                core_distances=model.core_distances_,
                                                ordering=model.ordering_, eps=0.5)
            if len(set(labels_050)) > 1:
                score_050 = silhouette_score(X, labels_050)
                if score_050 > best_score:
                    labels = labels_050
                    best_score = score_050
          
            labels_200 = cluster_optics_dbscan(reachability=model.reachability_,
                                                core_distances=model.core_distances_,
                                                ordering=model.ordering_, eps=2)
            if len(set(labels_200)) > 1:
                score_200 = silhouette_score(X, labels_200)
                if score_200 > best_score:
                    labels = labels_200
                    best_score = score_200
            if best_score == -1:
                return X, y

            not_noise_mask = (labels != -1)
            best_X_no_outliner = X[not_noise_mask]
            best_y_no_outliner = np.array(y)[not_noise_mask]      
            if best_y_no_outliner.shape[0] < min_num_samples: 
                return X, y  
     
        elif 'dbscan' in model_name:
            '''Picking the best model with neighborhoods containing more than 1/10, 1/15 or 1/20 of min_num_samples 
            and keep filtered data only if total number of samples is greater more than min_num_samples'''
            
            eps = [0.1, 0.2, 0.3]
            param_grid = {'eps': eps}
            dbscan_model = DBSCAN(n_jobs=-1)
            min_samples = [min_num_samples//10, min_num_samples//20, min_num_samples//15]
            for e in eps:
                for m in min_samples:
                    dbscan_model.set_params(eps=e, min_samples=m)
                    y_pred = dbscan_model.fit_predict(X)
                    score = self.silhouette_scorer(X, y_pred)
                    if score > best_score:
                        best_score = score
                        best_pred = y_pred
                        best_estimator = dbscan_model
                        
            if best_score == -1:
                return X, y  
             
            core_samples_mask = np.zeros_like(best_estimator.labels_, dtype=bool)
            core_samples_mask[best_estimator.core_sample_indices_] = True
            best_X_no_outliner = X[core_samples_mask]
            best_y_no_outliner = np.array(y)[core_samples_mask]
           
            if best_y_no_outliner.shape[0] > min_num_samples:
                not_noise_mask = (best_pred != -1)
                best_X_no_outliner = X[not_noise_mask]
                best_y_no_outliner = np.array(y)[not_noise_mask]
        else:
            return X, y  
     
        return best_X_no_outliner,  best_y_no_outliner
    
    
    def feature_analysis(self, n_components, feature_names, vectors, model_name='NMF'):
        '''Feature analysis to reduce number of feature to improve clustering accuracy'''
        
        print(n_components, vectors.shape)
        if model_name == 'PCA':
            pca_model = PCA()
            param_grid = {'n_components': [min(vectors.shape[1]-1, n_components-5), n_components, min(vectors.shape[1]-1, n_components +5)] , 'whiten': [True, False], 'svd_solver': ['auto', 'full', 'arpack', 'randomized']}
            grid_search = GridSearchCV(pca_model, param_grid=param_grid, n_jobs=-1)
           
            try:
                grid_search.fit(vectors)
            except:
                return vectors
              
            best_transformed_data = grid_search.best_estimator_.transform(vectors)
    
    
        elif model_name == 'NMF':
            beta_losses = ['frobenius', 'kullback-leibler']
            solvers = ['mu', 'cd']
            pca_model = NMF(n_components=n_components, random_state=1, max_iter=1000, alpha=.1, verbose=0, l1_ratio=.5)
           
            best_err = 1e+10
            for beta_loss in beta_losses:
                for solver in solvers:
                    if ('kullback' in beta_loss) or ('cd' in solver):
                        continue
                    pca_model.set_params(beta_loss=beta_loss, solver=solver)
                    transformed_data = pca_model.fit_transform(vectors)
                    err = pca_model.reconstruction_err_
                    if err < best_err:
                        best_err = err
                        best_transformed_data = transformed_data
            if self.printout == True:
                self.plot_top_words(pca_model.components_, model_name, feature_names)
            
                       
        elif model_name == 'LDA':
            pca_model = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        verbose=0, random_state=0, n_jobs=-1)
            lds = [0.5, 0.7, 0.9]
            best_perplexity = 1e+10
            for ld in lds:
                pca_model.set_params(learning_decay=ld)
                transformed_data = pca_model.fit_transform(vectors)
                perplexity = pca_model.perplexity(vectors)
                
                if best_perplexity > perplexity:
                    best_perplexity = perplexity
                    best_transformed_data =transformed_data
                    score = pca_model.score(vectors)
                    best_comp = pca_model.components_
            if self.printout == True:
                self.plot_top_words(best_comp, model_name, feature_names)
                   
        elif model_name == 'SVD':
     
            pca_model = TruncatedSVD(n_components, algorithm='randomized', n_iter=10, random_state=0)
            best_transformed_data = pca_model.fit_transform(vectors)
            normalizer = Normalizer(copy=False)
            transformed_data = normalizer.transform(best_transformed_data)
            best_comp = pca_model.components_
            if self.printout == True:
                self.plot_top_words(best_comp, model_name, feature_names)
       
        elif model_name == 'BernoulliRBM':
            batch_size = 32
            lrs = [0.01, 0.02]
            best_score = -1e+10
           
            for lr in lrs:
                pca_model = BernoulliRBM(n_components=n_components, learning_rate=lr, batch_size=batch_size, n_iter=10, verbose=0, random_state=1)
                transformed_data = pca_model.fit_transform(vectors)
                score_smp = pca_model.score_samples(vectors)
                print(pca_model.components_)
                
                score = sum(score_smp)/len(score_smp)
                if best_score < score:
                    best_score = score
                    best_transformed_data = transformed_data
                    best_comp = pca_model.components_
            if self.printout == True:
                self.plot_top_words(best_comp, model_name, feature_names)
        else:
            best_transformed_data = vectors

        return best_transformed_data
       
    
     
    def text_clustering(self):
        '''Main function to performe all required steps for unsupervised learning: 
            Vectorization, Dimentionality reduction, outliner removal, clustering''' 
            
        X_train, y_true = get_train_data(self.data_dir, min_text_length=self.min_text_length, 
                                          printout=self.printout)

        scores, prev_n_features = [], []
        print('Started training for %d samples using %s vectorizer \n'%(len(y_true), self.vec_model))

        for min_df in self.min_frequency:
            for max_df in self.max_frequency:
                gc.collect()
    
                feature_names, vectors, n_features = \
                    self.vectorizing(X_train, min_df, max_df, prev_n_features)
                
                if n_features == 0 or n_features < self.min_vocab_size:
                    continue
                
                prev_n_features.append(n_features)

                if vectors.shape[0] <= vectors.shape[1]:
                    #print(vectors.shape)
                    continue  
                
                '''Use PCA() to determine the number of features for 90% of explaine variance'''
                suggested_n_features = self.get_n_feature(vectors)
                
                if suggested_n_features >= n_features:
                    #print(suggested_n_features, n_features)
                    continue
                
                for dr_model  in self.dim_reduction_model:
                    t0 = time.time() 
                    X = self.feature_analysis(suggested_n_features, \
                                                         feature_names, vectors, model_name=dr_model)
                    if self.printout == True:
                        print('After {} total number of features left - {} out of {}'.\
                              format(dr_model.upper(), X.shape[1], vectors.shape[1]))
                    for or_model  in self.outliner_removal_model:
                        X_no_ouliners, y_no_ouliners = self.outliner_detection(or_model, X, y_true)
                        if self.printout == True:
                            print('After {} total number of samples left - {} out of {}'.format(or_model.upper(), X_no_ouliners.shape[0], X.shape[0]))
    
                        for cl_model  in self.clustering_model:
                            print(cl_model)
                            if str('Kmeans').lower() in cl_model.lower():
                                param_grid = {'n_clusters': np.arange(self.min_k, self.max_k)}
                                kmeans_model = MiniBatchKMeans(init='k-means++', batch_size=256)
                                scorer = make_scorer(self.silhouette_scorer)
                                grid_search = GridSearchCV(kmeans_model, param_grid=param_grid, scoring=scorer, n_jobs=-1)
                                grid_search.fit(X_no_ouliners, X_no_ouliners)
                                
                                best_estimator = grid_search.best_estimator_
                                y_pred = best_estimator.predict(X_no_ouliners)
                                score = grid_search.best_score_
                                        
                            elif str('GaussianMixture').lower() in cl_model.lower():
                                param_grid = {'n_components': np.arange(self.min_k, self.max_k), 'covariance_type': ['spherical', 'tied', 'diag', 'full']}
                                gm_model = BayesianGaussianMixture(max_iter=200, warm_start=True)
                                scorer = make_scorer(self.silhouette_scorer)
                                grid_search = GridSearchCV(gm_model, param_grid=param_grid, scoring=scorer, n_jobs=-1)
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    grid_search.fit(X_no_ouliners, X_no_ouliners)
             
                                best_estimator = grid_search.best_estimator_
                                y_pred = best_estimator.predict(X_no_ouliners)
                                score = grid_search.best_score_
    
                            else:
                                print('Supported Clustering model GaussianMixture or Kmeans, \
                                     please check your input')
                                continue
    
                            v_score = v_measure_score(y_no_ouliners, y_pred)
                            c_score = self.calinski_harabasz_scorer(X_no_ouliners, y_no_ouliners)
                            d_score = self.davies_bouldin_scorer(X_no_ouliners, y_no_ouliners)
                            best_n_clusters = best_estimator.get_params()['n_clusters']
        
                            scores.append((v_score, score, c_score, d_score, best_n_clusters))
                            if self.printout == True:    
                                print('Got {} clusters in {}s'.format(best_n_clusters, int(time.time() - t0)))
                                print('V score = {}, Silhouette score = {}, calinski_harabasz_score = {}; \
                                      davies_bouldin_score={}'.format(v_score, score, c_score, d_score))
                            
        if len(scores) > 0:
            v_scores, s_scores, c_scores, d_scores, n_clusters = zip(*scores)
            v_scores = np.max(np.array(v_scores))
            v_clusters = n_clusters[np.argmax(np.array(v_scores))]
            s_scores = np.max(np.array(s_scores)) 
            s_clusters = n_clusters[np.argmax(np.array(s_scores))]
            c_scores = np.max(np.array(c_scores)) 
            c_clusters = n_clusters[np.argmax(np.array(c_scores))]
            d_scores = np.max(np.array(d_scores)) 
            d_clusters = n_clusters[np.argmax(np.array(d_scores))]
            print('Best V-measure: {} for {} clusters'.format(v_scores, v_clusters))
            print('Best Silhouette Coefficient: {} for {} clusters'.format(s_scores, s_clusters))
            print('Best Calinski-Harabasz Index: {} for {} clusters'.format(c_scores, c_clusters))
            print('Best Davies-Bouldin Index: {} for {} clusters'.format(d_scores, d_clusters))
                
            
