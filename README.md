# ranking-kNN-imputation
An imputation method based on ranking and nearest neighbor (kNN) as described in:
    
Beretta L, Santaniello A. "Nearest neighbor imputation algorithms: a critical evaluation". BMC Med Inform Decis Mak. 2016 Jul 25;16 Suppl 3:74. https://doi.org/10.1186/s12911-016-0318-z


    The method imputes missing data combining a non linear feature selection and a nearest neighbor imputation algorithm.

    The algoroithm discards features that may be irrelevant or noisy for distance determination and hence for NN determination.

    The variable with missing instances is used as a class feature and ranking algorithms are used to select the variables that best correlate with the "class". The search is performed consdiering complete cases and works on a normalized (by span) dataset.

    Compared to the original algorithm described by Beretta et al., the code allows the automated selection of relevant features and optimizes the number k neighbors used for imputation.

    The elbow method is used to automatically select the parameters. Alternatively, the number of features and k neighobrs can be manually set.
        
        
Non-linear feature ranking algiorithm are recommended. 
By default the algorithm works using ReliefF-based methods.
The sklearn package Rebate (https://github.com/EpistasisLab/scikit-rebate) needs to be installed.
