'''
An imputation method based on ranking and nearest neighbor.
    
Beretta L, Santaniello A. Nearest neighbor imputation algorithms: a critical
evaluation. BMC Med Inform Decis Mak. 2016 Jul 25;16 Suppl 3:74.
'''
#Author: Lorenzo Beretta, lorberimm@hotmail.com

import numpy as np
from skrebate import ReliefF, MultiSURF                # The skrebate module needs to be installed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings


class Impute (object):
    """Impute missing data combining a non linear feature selection and nearest neighbor imputation.
    
    The algoroithm discards features that may be irrelevant or noisy for distance determination and 
        hence for NN determination.
    
    The variable with missing instances is used as a class feature and ranking algorithms are used to 
        select the variables that best correlate with the "class". The search is performed considering
        complete cases and works on normalized (by span) dataset.
        
    Compared to the original algorithm described by Beretta et al., the code allows the automated
        selection of relevant features and optimizes the number k neighbors used for imputation.
        The elbow method is used to automatically select the parameters.
        Alternatively, the number of features and k neighobrs can be manually set.
        
    
    Parameters
    ----------
    
    esitmator: estimator object
        An estimator object implementing "fit".
        Preferred choices:
            ReliefF ()
            MultiSURF ()                  --> default method
            RandomForestRegressor ()
    
    k: number of neighbors used for imputation
        k = None      : automatic choice of k      --> defaut method
        k = 1         : hot-deck method
        k > 1         : kNN method
        k > instances : mean imputation
    """
    
    
    def __init__ (self, estimator=MultiSURF (), k=None):
        self.estimator = estimator
        self.k = k
 
       
    def fit (self, X, Y=None):
        '''Impute missing data
        
        Parameters
        ----------

        X : {array-like, sparse matrix}, shape = [n_samples, n_features] 
            where, missing values are exclusively coded by "?"  
        
        
        Returns
        -------
        
        self
        
        '''  
        
        if np.any (X == "?") == False:
            raise ValueError("No missing values coded as (?) can be detected")
        
        return self
        
    
        
    def transform (self, X):
        """Impute missing values in X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The input data to complete.
            
    
        Returns
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The imputed dataset.
        
        """
        
        nInst, nFeat = np.shape (X)
        
        # Imputation procedure
        # find instances and features with missing or complete values
        missing_inst = np.where (np.any (X == "?", axis=1)) [0]   
        missing_feat = np.where (np.any (X == "?", axis=0)) [0] 
        complete_inst = []
        for ci in range (nInst):
            if ci not in missing_inst:
                complete_inst.append (ci)
        complete_feat = []
        for cf in range (nFeat):
            if cf not in missing_feat:
                complete_feat.append (cf)
                
        # normalize data
        scaler = MinMaxScaler ()
        n_X = X.copy ()
        for cf in range (nFeat):
            index_complete_cf = np.where (X [:, cf] != "?") [0]
            complete_cf = X [index_complete_cf, cf]
            complete_cf = complete_cf.astype (float)
            complete_cf = complete_cf.reshape (-1, 1)
            n_complete_cf = scaler.fit_transform (complete_cf).T
            n_X [index_complete_cf, cf] = n_complete_cf  
            
        # complete sets normalizez (cn) or not (c)
        cn_X = n_X [complete_inst, :]
        cn_X = cn_X.astype (float)
        c_X = X [complete_inst, :]
        c_X = c_X.astype (float)
        nComplete_inst = len (complete_inst)
        
        # iterate through features with missing values
        for cf in missing_feat:
            # Rank features: current as target; remaining as predictors
            # Not all the features will be used for imputation, just those
            #   that best predict the current feature
            af = []
            scores, ranked_feat = self.ranker (cn_X, cf, af, nFeat)
            
            # selected features (by elbow method)
            sel_feat = self.elbow (scores, ranked_feat)
            exCept = False
            
            # Iterate through missing values (current feature)
            missing_cf = np.where (n_X [:, cf] == "?") [0]
            for cm in missing_cf:
                origin = n_X [cm, :]
                
                # Check if instance is complete, if it contains several missing values
                #   variables should be re-ranked not considering the additional features (af) 
                #   with missing values (cannot contribute to distance and NN-imputation)
                cm_miss = np.where (origin == "?") [0]
                if cm_miss.shape [0] > 1:
                    index = np.where (cm_miss != cf) [0]
                    af = cm_miss [index]
                    af = list (af)
                    scores, ranked_feat = self.ranker (cn_X, cf, af, nFeat)
                    exCept = True
                    sel_feat_alt = self.elbow (scores, ranked_feat)
                    
                # NN procedure to impute missing values
                # Use the selected features (according to the elbow method)
                if exCept == True:
                    sel_feat_NN = sel_feat_alt
                else:
                    sel_feat_NN = sel_feat
                exCept = False
                
                # calculate distances
                red_origin = origin [sel_feat_NN] 
                red_origin = red_origin.reshape (1, -1)
                red_donors = cn_X [:, sel_feat_NN]
                distances = euclidean_distances (red_donors, red_origin)
                
                # order distances
                order = np.argsort(distances [:, 0])
                index = np.arange (nComplete_inst)
                NN = index [order]
                
                # order feature and find value (mean of k values)
                if self.k == None:              
                    distances = 1/distances
                    # calculate k via the elbow method
                    kNN = self.elbow (distances [order, 0], NN)
                else:    
                    if self.k > nComplete_inst:
                        kNN = NN [:nComplete_inst]
                    else:
                        kNN = NN [:self.k]
                imputed_value = np.mean (c_X [kNN, cf])
                X [cm, cf] = imputed_value
                
        # return the imputed dataset
        X_ = X.astype (float)
        return X_
    
    
    def ranker (self, cn_X, cf, af, nFeat):
        '''Ranks features according to different algorithms'''
        
        y = cn_X [:, cf]
        remain = np.arange (nFeat)
        af.append (cf)
        remain = np.delete (remain, af)
        x = cn_X [:, remain]        
        self.estimator.fit(x, y)
        scores = self.estimator.feature_importances_
        order = np.argsort(scores)[::-1]
        scores = scores [order]
        ranked_feat = remain [order]
        return scores, ranked_feat
    
    
    def elbow (self, scores, ranked_feat):
        '''Finds the optimal point via the elbow method (2nd derivative)'''
        
        # To be used to find the optimal number of features or k
        nValid = ranked_feat.shape [0]
        best = 0
        for cval in range (1, nValid -1):
            deriv2 = abs (scores [cval + 1] + scores [cval - 1] - 2 * scores [cval])
            if deriv2 > best:
                best = deriv2
                point = cval

        return ranked_feat [:point+1]
