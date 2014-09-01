function [trainSamples, testSamples, ...
    trainLabels_ar, trainLabels_ar_original,testLabels_ar, testLabels_ar_original,...
    trainLabels_val, trainLabels_val_original, testLabels_val, testLabels_val_original, ...
    centres_val, centres_ar, num_features] = Prepare_CNF_music_data(levels, num_folds_test)

[trainSamples_comb, testSamples_comb, trainLabels_ar_comb, ~, ...
testLabels_ar_comb, trainLabels_val_comb, ~, testLabels_val_comb,...
~, ~, ~, ~, num_features] = Prepare_SVR_music_data(num_folds_test, false);

% put the collected and normalised data into CCNF suitable format
seq_length = 15;

% initialising train stuff
trainSamples = cell(num_folds_test, 1);

% Quantised version of the signal
trainLabels_ar = cell(num_folds_test, 1);
trainLabels_val = cell(num_folds_test, 1);

% Non-quantised version
trainLabels_ar_original = cell(num_folds_test, 1);
trainLabels_val_original = cell(num_folds_test, 1);

% initialising test stuff
testSamples = cell(num_folds_test, 1);

% Quantised version of the signal
testLabels_ar = cell(num_folds_test, 1);
testLabels_val = cell(num_folds_test, 1);

% Non-quantised version
testLabels_ar_original = cell(num_folds_test, 1);
testLabels_val_original = cell(num_folds_test, 1);

centres_val = zeros(num_folds_test, levels);
centres_ar = zeros(num_folds_test, levels);

for fold=1:num_folds_test
            
    % do the training data
    num_sequences_train = size(trainSamples_comb{fold},1)/seq_length;
    
    % samples
    train_samples_fold = cell(num_sequences_train, 1);
    
    % labels
    train_labels_ar_fold = cell(num_sequences_train, 1);
    train_labels_val_fold = cell(num_sequences_train, 1);
        
    for i=1:num_sequences_train
        beg_ind = (i-1)*seq_length + 1;
        end_ind = i*seq_length;
        train_samples_fold{i} = trainSamples_comb{fold}(beg_ind:end_ind,:)';
        train_labels_ar_fold{i} = trainLabels_ar_comb{fold}(beg_ind:end_ind)';
        train_labels_val_fold{i} = trainLabels_val_comb{fold}(beg_ind:end_ind)';        
    end
    
    trainLabels_ar_original{fold} = train_labels_ar_fold;
    trainLabels_val_original{fold} = train_labels_val_fold;
    
    % extract the quantisation labels
    min_ar = min(cell2mat(train_labels_ar_fold'));
    max_ar = max(cell2mat(train_labels_ar_fold'));
    
    steps_ar = (max_ar - min_ar) / levels;
    
    centres_ar(fold,:) = min_ar + steps_ar / 2 : steps_ar : max_ar;
    
    min_val = min(cell2mat(train_labels_val_fold'));
    max_val = max(cell2mat(train_labels_val_fold'));
    
    steps_val = (max_val - min_val) / levels;
    
    centres_val(fold,:) = min_val + steps_val / 2 : steps_val : max_val;
    
    % quantise the sequences
    ar_classes = bsxfun(@plus, cell2mat(train_labels_ar_fold'), -centres_ar(fold,:)');
    [~,ar_classes] = min(abs(ar_classes)); 
    ar_classes = int32(ar_classes) - 1;% so that indices start at 0
    
%     figure;
%     plot(cell2mat(train_labels_ar_fold'));
%     hold on;
%     plot((min_ar + steps_ar / 2) + steps_ar .* double(ar_classes'), 'r');

    train_labels_ar_fold = mat2cell(ar_classes, 1, seq_length * ones(num_sequences_train,1))';

    val_classes = bsxfun(@plus, cell2mat(train_labels_val_fold'), -centres_val(fold,:)');
    [~,val_classes] = min(abs(val_classes));
    val_classes = int32(val_classes) - 1;% so that indices start at 0
    
%     figure;
%     plot(cell2mat(train_labels_val_fold'));
%     hold on;
%     plot((min_val + steps_val / 2) + steps_val .* double(val_classes), 'r');

    train_labels_val_fold = mat2cell(val_classes, 1, seq_length * ones(num_sequences_train,1))';

    trainSamples{fold} = train_samples_fold;

    trainLabels_ar{fold} = train_labels_ar_fold;
    trainLabels_val{fold} = train_labels_val_fold;

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
        test_samples_fold{i} = testSamples_comb{fold}(beg_ind:end_ind,:)';
        test_labels_ar_fold{i} = testLabels_ar_comb{fold}(beg_ind:end_ind)';
        test_labels_val_fold{i} = testLabels_val_comb{fold}(beg_ind:end_ind)';
    end
    
    testSamples{fold} = test_samples_fold;

    testLabels_ar_original{fold} = test_labels_ar_fold;
    testLabels_val_original{fold} = test_labels_val_fold;
    
    % quantise the sequences
    ar_classes = bsxfun(@plus, cell2mat(test_labels_ar_fold'), -centres_ar(fold,:)');
    [~,ar_classes] = min(abs(ar_classes));
    ar_classes = int32(ar_classes) - 1; % so that indices start at 0
    
%     figure;
%     plot(cell2mat(test_labels_ar_fold'));
%     hold on;
%     plot((min_ar + steps_ar / 2) + steps_ar .* double(ar_classes), 'r');

    test_labels_ar_fold = mat2cell(ar_classes, 1, seq_length * ones(num_sequences_test,1))';
    
    val_classes = bsxfun(@plus, cell2mat(test_labels_val_fold'), -centres_val(fold,:)');
    [~,val_classes] = min(abs(val_classes));
    val_classes = int32(val_classes) - 1; % so that indices start at 0

%     figure;
%     plot(cell2mat(test_labels_val_fold'));
%     hold on;
%     plot((min_val + steps_val / 2) + steps_val .* double(val_classes), 'r');

    test_labels_val_fold = mat2cell(val_classes, 1, seq_length * ones(num_sequences_test,1))';

    testLabels_ar{fold} = test_labels_ar_fold;
    testLabels_val{fold} = test_labels_val_fold;
end

end