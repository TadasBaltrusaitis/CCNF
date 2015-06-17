clear
clc

num_alphas=10;
input_layer_size=100;

an=randInitializeWeights({[num_alphas*2,input_layer_size],[num_alphas,num_alphas]});