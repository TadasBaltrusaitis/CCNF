function Script_CRF

addpath(genpath('../hCRF'));

num_test_folds = 5;
num_train_folds = 2;

correlations_ar = cell(num_test_folds, 1);
correlations_val = cell(num_test_folds, 1);
long_correlations_ar = zeros(num_test_folds,1);
long_correlations_val = zeros(num_test_folds,1);

RMS_ar = cell(num_test_folds, 1);
RMS_val = cell(num_test_folds, 1);
long_RMS_ar = zeros(num_test_folds,1);
long_RMS_val = zeros(num_test_folds,1);

euclidean = zeros(num_test_folds, 1);

predictions_ar = cell(num_test_folds, 1);
predictions_val = cell(num_test_folds, 1);

gt_ar = cell(num_test_folds, 1);
gt_val = cell(num_test_folds, 1);

% The number of quantisation levels
num_levels = 7;

% load the training and testing data

% TODO add bias?

[trainSamples, testSamples, ...
    trainLabels_ar, trainLabels_ar_original,testLabels_ar, testLabels_ar_original,...
    trainLabels_val, trainLabels_val_original, testLabels_val, testLabels_val_original, ...
    centres_val, centres_ar, num_features] = Prepare_CNF_music_data(num_levels, num_test_folds);


%%
sequence_length = 15;

% Set up the hyperparameters to be validated
hyperparams.caption = 'CRF';
hyperparams.optimizer = 'bfgs';
hyperparams.windowSize = 0;
hyperparams.modelType = 'crf';
hyperparams.maxIterations = 300;
hyperparams.nbIterations = 1;
hyperparams.debugLevel = 0;
hyperparams.computeTrainingError = 1;
hyperparams.rangeWeights = [0,0];
hyperparams.regFactor = [0, 0.01, 1, 10, 100, 1000];

hyperparams.validate_params = {'regFactor'};

% Set the training function
train = @crf_train;
    
% Set the test function (the first output will be used for validation)
test = @crf_test;

tic
for test_fold = 1:num_test_folds

    % Set the testing centres (for converting from quantised classes
    % values)


    % Create the crossvalidation indices here (in order to make sure that
    % samples from same song don't go to both train and test folds)
    num_cv_folds = 2;    

    % 
    num_sequences = size(trainSamples{test_fold}, 1);

    [ inds_cv_train, ~ ] = GetFolds(num_sequences, num_cv_folds);
    
    hyperparams.centres = centres_ar(test_fold,:);
        
    % Crossvalidate arousal model
    [ best_params_ar, all_params_ar ] = crossvalidate_grid_search(train, test, true, trainSamples{test_fold}, trainLabels_ar{test_fold}, hyperparams, 'inds_cv_train', inds_cv_train);
    
    hyperparams.centres = centres_val(test_fold,:);
    
    % Crossvalidate valence model
    [ best_params_val, all_params_val ] = crossvalidate_grid_search(train, test, true, trainSamples{test_fold}, trainLabels_ar{test_fold}, hyperparams, 'inds_cv_train', inds_cv_train);
    % Flatten the training data
    
    best_params(test_fold).arousal = best_params_ar;
    best_params(test_fold).valence = best_params_val;
    
    %% The actual testing results
    model_ar = train(trainLabels_ar{test_fold}, trainSamples{test_fold}, best_params_ar);
    model_val = train(trainLabels_val{test_fold}, trainSamples{test_fold}, best_params_val);

    [~, prediction_ar] = test(testLabels_ar{test_fold}, testSamples{test_fold}, model_ar);
    [~, prediction_val] = test(testLabels_val{test_fold}, testSamples{test_fold}, model_val);
    
    % Convert from classes to double values
    step_ar = model_ar.centres(2) - model_ar.centres(1);
    min_ar = model_ar.centres(1) - step_ar/2;    
    prediction_ar = (min_ar + step_ar / 2) + step_ar .* double(prediction_ar');
    
    step_val = model_val.centres(2) - model_val.centres(1);
    min_val = model_val.centres(1) - step_val/2;        
    prediction_val = (min_val + step_val / 2) + step_val .* double(prediction_val');
    
    predictions_ar{test_fold} = prediction_ar';
    predictions_val{test_fold} = prediction_val';
    
    gt_ar{test_fold} = cell2mat(testLabels_ar_original{test_fold}')';
    gt_val{test_fold} = cell2mat(testLabels_val_original{test_fold}')';    
    
    [ correlations_ar{test_fold}, RMS_ar{test_fold},...
      long_correlations_ar(test_fold), long_RMS_ar(test_fold) ] = Evaluate_music_predictions(gt_ar{test_fold}, predictions_ar{test_fold});
    
    [ correlations_val{test_fold}, RMS_val{test_fold},...
        long_correlations_val(test_fold), long_RMS_val(test_fold) ] = Evaluate_music_predictions(gt_val{test_fold}, predictions_val{test_fold});       
            
    % Normalise the input labels to be 0-1, do the same to the
    % output ones, so that euclidean distances are the
    % comparable to other work    
    offset_ar = -1;
    scaling_ar = 2;
    
    prediction_normed_ar = (predictions_ar{test_fold} - offset_ar)/scaling_ar;
    test_labels_normed_ar = (gt_ar{test_fold} - offset_ar)/scaling_ar;    

    offset_val = -1;
    scaling_val = 2;
    
    prediction_normed_val = (predictions_val{test_fold} - offset_val)/scaling_val;
    test_labels_normed_val = (gt_val{test_fold} - offset_val)/scaling_val;

    euclidean(test_fold) = mean(sqrt((prediction_normed_ar - test_labels_normed_ar).^2 + ...
                           (prediction_normed_val - test_labels_normed_val).^2));      
                       
end

time_taken = toc;
short_correlations_ar = mean(cat(2,correlations_ar{:}));
short_correlations_val = mean(cat(2,correlations_val{:}));

short_rms_ar = mean(cat(1,RMS_ar{:}));
short_rms_val = mean(cat(1,RMS_val{:}));

save('results/CRF_res', 'correlations_ar', 'RMS_ar', 'short_correlations_ar', 'long_correlations_ar', 'long_RMS_ar', 'short_rms_ar', ...
                 'correlations_val', 'RMS_val', 'short_correlations_val', 'long_correlations_val', 'long_RMS_val', 'short_rms_val',...
                 'time_taken', 'euclidean', 'predictions_ar', 'predictions_val', 'gt_ar', 'gt_val');
end

function [model] = crf_train(train_labels, train_samples, hyper)
    
    [model, stats] = trainCRF(train_samples, train_labels, hyper);
    
    model.centres = hyper.centres;
end

function [result, prediction] = crf_test(test_labels, test_samples, model)
        
    % predict
    [lhoods_pred, ~] = testCRF(model, test_samples, test_labels);
    
    [~, labels] = max(cell2mat(lhoods_pred));
    labels = labels - 1;
    
    prediction_reshaped = reshape(labels, 15, numel(labels)/15);
    test_labels_reshaped = double(cell2mat(test_labels)');
    
    % Will be optimising 'rmse' like error (don't want accuracy as that
    % does not account for distance between classes)
    rms_per_song = sqrt(mean((prediction_reshaped - test_labels_reshaped).^2));
    result = mean(rms_per_song);  
        
    prediction = labels';
end