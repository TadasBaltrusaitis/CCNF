To recreate the AU recognition results in the "Continuous Conditional Neural Fields for
Structured Regression" paper:

1. You will first need to acquire the DISFA dataset http://www.engr.du.edu/mmahoor/DISFA.htm
2. Modify the shared_defs folder with the location of the DISFA dataset
3. For SVR and CCRF experiments, download and compile the liblinear library http://www.csie.ntu.edu.tw/~cjlin/liblinear/, currently it's assumed it's in "C:/liblinear/"
4. Run:
    Script_CCNF.m
    Script_SVR.m
    Script_SVR_CCRF.m

The first script you run will extract the appearance features and place them in DISFA directory (over 1GB), so make sure there is enough space there.
This will make the first script to take some time to start machine learning, as Matlab is quite slow with reading video files.

Note results will vary slightly based on the version of Matlab you use and the operating system (numerical issues).

These experiments take some time to run due to the need to train and validate 12 action unit models for every 27 of subjects (leading to validation and training of 324 models), this involves training dimensionality reduction and the models themselves. All three experiments can take up to a week on a single machine.

Results I got on Matlab2012b and Windows 8 machine can be found in results folder.
