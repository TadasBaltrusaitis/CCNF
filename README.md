These files contain the libraries needed to train and test Continuous Conditional Neural Fields (CCNF) and Continuous Conditional Random Fields (CCRF).

The project was tested on Matlab R2012b and R2013a (can't guarantee compatibility with other versions).

Some of the experiments rely on the availability of mex compiled liblinear (http://www.csie.ntu.edu.tw/~cjlin/liblinear/) and libsvm (http://www.csie.ntu.edu.tw/~cjlin/libsvm/) on your machine.

--------------------------------------- Copyright information -----------------------------------------------------------------------------	

Copyright can be found in the Copyright.txt

--------------------------------------- Code Layout -----------------------------------------------------------------------------


./CCNF - the training and inference libraries for CCNF
./CCRF - the training and inference libraries for CCRF

./music_emotion - emotion in music prediction experiments, comparing the use of CCNF, CCRF, Neural Net (CCNF without edge), and SVR models
    results/ - the results from running the experiments

./patch_experts - training code for patch expert training (for facial landmark detection), more in the readme.txt in the relevant folder
    ccnf_training/ - training CCNF patch experts (for the Constrained Local Neural Fields for robust facial landmark detection in the wild paper)
    data_preparation/ - converting image and landmark datasets to the right formats
    svr_training/ - training SVR patch experts (the standard CLM patch experts)

--------------------------------------- Final remarks -----------------------------------------------------------------------------	

I did my best to make sure that the code runs out of the box but there are always issues and I would be grateful for your understanding that this is research code and not a commercial
level product. However, if you encounter any probles please contact me at Tadas.Baltrusaitis@cl.cam.ac.uk for any bug reports/questions/suggestions. 
