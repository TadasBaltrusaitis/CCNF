function Script_CNF

addpath(genpath('../hCRF'));

num_test_folds = 27;

correlations = zeros(num_test_folds, 1);
RMS = zeros(num_test_folds, 1);

predictions = cell(num_test_folds, 1);

gt = cell(num_test_folds, 1);

%%
% Set up the hyperparameters to be validated
hyperparams.caption = 'CNF';
hyperparams.optimizer = 'lbfgs';
hyperparams.windowSize = 0;
hyperparams.modelType = 'cnf';
hyperparams.maxIterations = 200;
hyperparams.nbIterations = 1;
hyperparams.rocRange = 1000;
hyperparams.debugLevel = 0;
hyperparams.computeTrainingError = 1;
hyperparams.rangeWeights = [-1,1];
hyperparams.normalizeWeights = [1];
hyperparams.regFactor = [0, 0.01, 1, 10, 100, 1000];
hyperparams.nbGates = [1, 5, 10, 20];

hyperparams.validate_params = {'regFactor', 'nbGates'};

% Set the training function
train = @cnf_train;
    
% Set the test function (the first output will be used for validation)
test = @cnf_test;

%% load shared definitions
shared_defs;

users = vid_user';

tic
for a=1:numel(aus)
    
    au = aus(a);    

    for test_fold = 1:num_test_folds
        
        %% use all but test_fold
        train_users = test_fold;
        test_users = setdiff(1:num_test_folds, test_fold);
        rest_aus = setdiff(aus, au);

        [data_train, labels_train, data_valid, labels_valid, data_test, labels_test, ~, seq_length] = collect_training_data(data_appearance, data_geom, patches_around_landmarks, vid_id, au_patches{a}, au, rest_aus, users, test_users, train_users, out_dir);
        
        %         
        data_train = data_train';
        data_valid = data_valid';
        data_test = data_test';
        tic

        num_features = size(data_train,2);
        
        data_train = mat2cell(data_train, seq_length*ones(1, size(data_train,1)/seq_length), num_features);        
        data_valid = mat2cell(data_valid, seq_length*ones(1, size(data_valid,1)/seq_length), num_features);        
        labels_train = mat2cell(labels_train, seq_length*ones(1, size(labels_train,1)/seq_length), 1);
        labels_valid = mat2cell(labels_valid, seq_length*ones(1, size(labels_valid,1)/seq_length), 1);
        
        data_test = {data_test};
        labels_test = {labels_test};
 
        % Validate the model
        [ best_params_fold, ~ ] = validate_grid_search(train, test, true, data_train, labels_train, data_valid, labels_valid, hyperparams, 'num_repeat', 2);

        best_params(test_fold).arousal = best_params_fold;

        %% The actual testing results
        
        % Combine train and validation for final training
        data_train = cat(1, data_train, data_valid);
        labels_train = cat(1, labels_train, labels_valid);
                
        model = train(labels_train, data_train, best_params_fold);

        [~, prediction] = test(labels_test, data_test, model);

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
end

time_taken = toc;
short_correlations_ar = mean(cat(2,correlations_ar{:}));
short_correlations_val = mean(cat(2,correlations_val{:}));

short_rms_ar = mean(cat(1,RMS_ar{:}));
short_rms_val = mean(cat(1,RMS_val{:}));

save('results/CNF_res', 'correlations_ar', 'RMS_ar', 'short_correlations_ar', 'long_correlations_ar', 'long_RMS_ar', 'short_rms_ar', ...
                 'correlations_val', 'RMS_val', 'short_correlations_val', 'long_correlations_val', 'long_RMS_val', 'short_rms_val',...
                 'time_taken', 'euclidean', 'predictions_ar', 'predictions_val', 'gt_ar', 'gt_val');
end

function [model] = cnf_train(train_labels, train_samples, hyper)
    
    [model, stats] = trainCNF(train_samples, train_labels, hyper);
    
    model.centres = hyper.centres;
end

function [result, prediction] = cnf_test(test_labels, test_samples, model)
        
    % predict
    [lhoods_pred, ~] = testCNF(model, test_samples, test_labels);
    
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