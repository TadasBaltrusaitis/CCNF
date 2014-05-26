function [trainSamples, testSamples, trainLabels_ar, trainLabels_ar_normed, testLabels_ar, trainLabels_val, trainLabels_val_normed, testLabels_val, offsets_val, offsets_ar, scalings_val, scalings_ar, num_features] = Prepare_SVR_music_data_per_frame(num_folds_test, add_bias)

    addpath('C:\libsvm\matlab');

    [label_vector_ar, instance_matrix_ar] = libsvmread('data/svmTrainAll-basic-arousal.libsvm'); 
    [label_vector_val, instance_matrix_val] = libsvmread('data/svmTrainAll-basic-valence.libsvm'); 
    instance_matrix_ar = full(instance_matrix_ar);
    label_vector_ar = full(label_vector_ar);
    label_vector_val = full(label_vector_val);

    %%
    sampleLength = 15;
    num_features = size(instance_matrix_ar,2);

    numSamples = numel(label_vector_ar)/sampleLength;

    % get the train (crossvalidation) and the test sets
    [ indsTrain, indsTest ] = GetFolds(numSamples, num_folds_test);

    trainSamples = cell(num_folds_test, 1);
    testSamples = cell(num_folds_test, 1);
    
    trainLabels_ar = cell(num_folds_test, 1);
    trainLabels_val = cell(num_folds_test, 1);
    
    testLabels_ar = cell(num_folds_test, 1);
    testLabels_val = cell(num_folds_test, 1);
    
    % the normed labels are needed as SVR predictions (or other
    % classifiers) are more stable when normalised (can later move back to
    % original using the scalings
    
    trainLabels_ar_normed = cell(num_folds_test, 1);
    trainLabels_val_normed = cell(num_folds_test, 1);    
    offsets_val = cell(num_folds_test, 1);
    offsets_ar = cell(num_folds_test, 1);
    scalings_val = cell(num_folds_test, 1);
    scalings_ar = cell(num_folds_test, 1);
    
    % first split the music data into the folds for training and testing
    for f=1:num_folds_test
        
        inds_train_fold = indsTrain(:,f);
        inds_test_fold = indsTest(:,f);
        
        size_train_fold = sum(inds_train_fold) * sampleLength;
        size_test_fold = sum(inds_test_fold) * sampleLength;
        
        trainSamples_fold = zeros(size_train_fold, num_features);
        testSamples_fold = zeros(size_test_fold, num_features);

        trainLabels_ar_fold = zeros(size_train_fold,1);
        trainLabels_val_fold = zeros(size_train_fold,1);
        
        testLabels_ar_fold = zeros(size_test_fold,1);
        testLabels_val_fold = zeros(size_test_fold,1);

        countTrain = 1;
        countTest = 1;

        for i=1:numSamples
            sample = instance_matrix_ar((i-1)*sampleLength+1:(i)*sampleLength,:);
            
            labelAr = label_vector_ar((i-1)*sampleLength+1:(i)*sampleLength);
            labelVal =  label_vector_val((i-1)*sampleLength+1:(i)*sampleLength);
            
            if(inds_train_fold(i))
                trainSamples_fold((countTrain-1)*sampleLength+1:(countTrain)*sampleLength,:) = sample;
                trainLabels_ar_fold((countTrain-1)*sampleLength+1:(countTrain)*sampleLength) = labelAr;
                trainLabels_val_fold((countTrain-1)*sampleLength+1:(countTrain)*sampleLength) = labelVal;
                countTrain = countTrain + 1;        
            else
                testSamples_fold((countTest-1)*sampleLength+1:(countTest)*sampleLength,:) = sample;
                testLabels_ar_fold((countTest-1)*sampleLength+1:(countTest)*sampleLength) = labelAr;
                testLabels_val_fold((countTest-1)*sampleLength+1:(countTest)*sampleLength) = labelVal;

                countTest = countTest + 1;        
            end
        end
        % testSamples = sparse(testSamples);
        % trainSamples = sparse(trainSamples);

        %% Data normalisation here
        offsets = min(trainSamples_fold,[],1);
        scalings = 1./(max(trainSamples_fold,[],1)-min(trainSamples_fold,[],1));

        normalisation = diag(scalings');
        normalisation(normalisation==inf) = 1;

        % normalising from 0 to 1
        trainSamples_fold = (trainSamples_fold - repmat(offsets,size(trainSamples_fold,1),1))*normalisation;        
        testSamples_fold = (testSamples_fold -repmat(offsets,size(testSamples_fold,1),1))*normalisation;

        % adding the bias term (unnecessary for CCNF)
        if(add_bias)            
            trainSamples_fold = cat(2, trainSamples_fold, ones(size(trainSamples_fold, 1),1));
            testSamples_fold = cat(2, testSamples_fold, ones(size(testSamples_fold, 1),1));
        end

        trainSamples{f} = full(trainSamples_fold);
        testSamples{f} = full(testSamples_fold);

        trainLabels_ar{f} = full(trainLabels_ar_fold);
        testLabels_ar{f} = full(testLabels_ar_fold);

        trainLabels_val{f} = full(trainLabels_val_fold);
        testLabels_val{f} = full(testLabels_val_fold);
        
        % the normed labels are needed as SVR predictions (or other
        % classifiers) are more stable when normalised (can later move back to
        % original using the scalings (not sure if SVR needs this?)
        offsets_val{f} = min(trainLabels_val{f});
        offsets_ar{f} = min(trainLabels_ar{f});
        
        scalings_val{f} = 1/(max(trainLabels_val{f}) - offsets_val{f});
        scalings_ar{f} = 1/(max(trainLabels_ar{f}) - offsets_ar{f});
        
        trainLabels_val_normed{f} = (trainLabels_val{f} -  offsets_val{f}) * scalings_val{f};
        trainLabels_ar_normed{f} = (trainLabels_ar{f} -  offsets_ar{f}) * scalings_ar{f};
        
%         for i=1:numel(trainLabels_val_normed_fold)
%             trainLabels_val_normed{i} = (trainLabels_val_fold{i} - offset_val) * scaling_val;
%             trainLabels_ar_normed{i} = (trainLabels_ar_fold{i} - offset_ar) * scaling_ar;
%         end
%         
    end            
    % for the bias term
    if(add_bias)
        num_features = num_features + 1;
    end
end