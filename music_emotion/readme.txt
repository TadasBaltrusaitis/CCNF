This code is in support of the "CCNF for continuous emotion tracking in music" paper by Vaiva Imbrasaite, Tadas Baltrusaitis, and Peter Robinson

This code provides the code for training and testing emotion in music prediction models.

The relevant dataset is packaged, you only need mex compiled liblinear (http://www.csie.ntu.edu.tw/~cjlin/liblinear/) and libsvm (http://www.csie.ntu.edu.tw/~cjlin/libsvm/) for experiment recreation.

To run the experiments use:
Script_CCNF.m trains and tests a linear-chain CCNF
Script_CCNF_no_edge.m trains and tests an unconnected CCNF (basically a Neural Network)
Script_SVR_CCRF.m trains and tests a CCRF model that relies on SVR-rbf predictions
Script_SVR_linear.m trains and tests a linear SVR model
Script_SVR_rbf.m trains and tests an RBF kernel SVR model