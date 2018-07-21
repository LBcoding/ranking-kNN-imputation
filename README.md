# ranking-kNN-imputation
An imputation method based on ranking and nearest neighbor (kNN) as described in:
    
Beretta L, Santaniello A. "Nearest neighbor imputation algorithms: a critical evaluation". BMC Med Inform Decis Mak. 2016 Jul 25;16 Suppl 3:74. https://doi.org/10.1186/s12911-016-0318-z


    The method imputes missing data combining a non linear feature selection and a nearest neighbor 
        imputation algorithm.

    The algoroithm discards features that may be irrelevant or noisy for distance determination and hence for 
        NN determination.

    The variable with missing instances is used as a class feature and ranking algorithms are used to select 
        the variables that best correlate with the "class". The search is performed consdiering complete cases 
        and works on a normalized (by span) dataset.

    Compared to the original algorithm described by Beretta et al., the code allows the automated selection of 
        relevant features and optimizes the number k neighbors used for imputation.

    The elbow method is used to automatically select the parameters. Alternatively, the number of features and 
        k neighobrs can be manually set.
        
        
Non-linear feature ranking algorithm are recommended to select relevant features.
By default the algorithm works using ReliefF-based methods.
The sklearn package Rebate (https://github.com/EpistasisLab/scikit-rebate) needs to be installed.


# Usage
The algorithm is designed to be integrated directly into scikit-learn machine learning workflows. ReliefF algorithms are used as feature selection step to optimize the search of closest neighbors.


    import numpy as np
    from skrebate import MultiSURF
    from Imputer import Impute

    dummy = np.array ((["0.2", "0.4", "5", "0.4", "6"],
                    ["?", "0.4", "8", "0.5", "8"],
                    ["0.1", "0.8", "3", "0.2", "6"],
                    ["0.3", "0.2", "?", "0.1", "7"],
                    ["0.2", "0.9", "3", "0.2", "4"]))

    imp = Impute (MultiSURF(), k=1)
    imp.fit (dummy)
    imputed_dummy = imp.transform (dummy)

    print (dummy)
    print ("")
    print (imputed_dummy)


    
