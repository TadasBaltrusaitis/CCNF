function [ indsTrain, indsTest ] = GetFolds( numSamples, numFolds )
%GETFOLDS Get indices for folds that will allow us to have equivalent
%cross-validations for different ML methods
%   Detailed explanation goes here

    rng(0);
    inds = crossvalind('Kfold',numSamples,numFolds);

    indsTrain = false(numSamples, numFolds);
    for i=1:numFolds
        indsTrain(:,i) = (inds~=i);        
    end
    indsTest = ~indsTrain;
    
end

