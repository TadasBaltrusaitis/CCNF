function [trainSamples, testSamples, trainLabels_ar, trainLabels_ar_normed,...
testLabels_ar, trainLabels_val, trainLabels_val_normed, testLabels_val,...
offsets_val, offsets_ar, scalings_val, scalings_ar, num_features] = Prepare_CCNF_music_data(num_folds_test)
   
[trainSamples_comb, testSamples_comb, trainLabels_ar_comb, trainLabels_ar_normed_comb,...
testLabels_ar_comb, trainLabels_val_comb, trainLabels_val_normed_comb, testLabels_val_comb,...
offsets_val, offsets_ar, scalings_val, scalings_ar, num_features]...
    = Prepare_SVR_music_data(num_folds_test, false);

% put the collected and normalised data into CCNF suitable format
seq_length = 15;

% initialising train stuff
trainSamples = cell(num_folds_test, 1);

trainLabels_ar = cell(num_folds_test, 1);
trainLabels_val = cell(num_folds_test, 1);

trainLabels_ar_normed = cell(num_folds_test, 1);
trainLabels_val_normed = cell(num_folds_test, 1);

% initialising test stuff
testSamples = cell(num_folds_test, 1);

testLabels_ar = cell(num_folds_test, 1);
testLabels_val = cell(num_folds_test, 1);


for fold=1:num_folds_test
    % do the training data
    num_sequences_train = size(trainSamples_comb{fold},1)/seq_length;
    
    % samples
    train_samples_fold = cell(num_sequences_train, 1);
    
    % labels
    train_labels_ar_fold = cell(num_sequences_train, 1);
    train_labels_val_fold = cell(num_sequences_train, 1);
    
    % normed labels
    train_labels_ar_normed_fold = cell(num_sequences_train, 1);
    train_labels_val_normed_fold = cell(num_sequences_train, 1);
    
    for i=1:num_sequences_train
        beg_ind = (i-1)*seq_length + 1;
        end_ind = i*seq_length;
        train_samples_fold{i} = trainSamples_comb{fold}(beg_ind:end_ind,:);
        train_labels_ar_fold{i} = trainLabels_ar_comb{fold}(beg_ind:end_ind);
        train_labels_val_fold{i} = trainLabels_val_comb{fold}(beg_ind:end_ind);
        train_labels_ar_normed_fold{i} = trainLabels_ar_normed_comb{fold}(beg_ind:end_ind);
        train_labels_val_normed_fold{i} = trainLabels_val_normed_comb{fold}(beg_ind:end_ind);
    end
    
    trainSamples{fold} = train_samples_fold;

    trainLabels_ar{fold} = train_labels_ar_fold;
    trainLabels_val{fold} = train_labels_val_fold;

    trainLabels_ar_normed{fold} = train_labels_ar_normed_fold;
    trainLabels_val_normed{fold} = train_labels_val_normed_fold;

    % do the test data
    num_sequences_test = size(testSamples_comb{fold},1)/seq_length;
    
    % samples
    test_samples_fold = cell(num_sequences_test, 1);
    
    % labels
    test_labels_ar_fold = cell(num_sequences_test, 1);
    test_labels_val_fold = cell(num_sequences_test, 1);
    
    for i=1:num_sequences_test
        beg_ind = (i-1)*seq_length + 1;
        end_ind = i*seq_length;
        test_samples_fold{i} = testSamples_comb{fold}(beg_ind:end_ind,:);
        test_labels_ar_fold{i} = testLabels_ar_comb{fold}(beg_ind:end_ind);
        test_labels_val_fold{i} = testLabels_val_comb{fold}(beg_ind:end_ind);
    end
    
    testSamples{fold} = test_samples_fold;

    testLabels_ar{fold} = test_labels_ar_fold;
    testLabels_val{fold} = test_labels_val_fold;
end

end