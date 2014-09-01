function Script_GP_new

addpath(genpath('../GP'));

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

% load the training and testing data
[trainSamples, testSamples, trainLabels_ar, trainLabels_ar_normed,...
    testLabels_ar, trainLabels_val, trainLabels_val_normed, testLabels_val,...
    offsets_val, offsets_ar, scalings_val, scalings_ar, num_features] = Prepare_CCNF_music_data(num_test_folds);

% 
num_sequences = zeros(num_test_folds, 1);

add_bias = true;

% Convert the sequences to independent samples
for test_fold = 1:num_test_folds
    
    num_sequences(test_fold) = size(trainLabels_ar{test_fold},1);
    
    trainLabels_ar{test_fold} = cell2mat(trainLabels_ar{test_fold});
    trainLabels_val{test_fold} = cell2mat(trainLabels_val{test_fold});
    
    testLabels_ar{test_fold} = cell2mat(testLabels_ar{test_fold});
    testLabels_val{test_fold} = cell2mat(testLabels_val{test_fold});
    
    trainSamples{test_fold} = cell2mat(trainSamples{test_fold});
    testSamples{test_fold} = cell2mat(testSamples{test_fold});
    
    if(add_bias)
        % Add bias
        trainSamples{test_fold} = cat(2, ones(size(trainSamples{test_fold},1), 1), trainSamples{test_fold});
        testSamples{test_fold} = cat(2, ones(size(testSamples{test_fold},1), 1), testSamples{test_fold});
    end
end

%%
sequence_length = 15;

% Set up the hyperparameters to be validated
hyperparams.lik = {'likGauss'};
hyperparams.inf = {'infExact'}; 

hyperparams.validate_params = {'sf', 'ell', 'sn'};
hyperparams.sf = [1, 2];
hyperparams.ell = [0.2, 0.4];   
hyperparams.sn = [0.1, 0.2, 0.5, 1];

% Set the training function
train = @gp_train;
    
% Set the test function (the first output will be used for validation)
test = @gp_test;

tic
for test_fold = 1:num_test_folds

    % Create the crossvalidation indices here (in order to make sure that
    % samples from same song don't go to both train and test folds)
    num_cv_folds = 2;    
    
    [ inds_cv_train, ~ ] = GetFolds(num_sequences(test_fold), num_cv_folds);
    
    inds_cv_train = (inds_cv_train(:,1) * ones(1, sequence_length))';
    inds_cv_train = logical(inds_cv_train(:));
    inds_cv_train = cat(2, inds_cv_train, ~inds_cv_train);
    
    % Crossvalidate arousal model
    [ best_params_ar, all_params_ar ] = crossvalidate_grid_search(train, test, true, trainSamples{test_fold}, trainLabels_ar{test_fold}, hyperparams, 'inds_cv_train', inds_cv_train);
    
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
    
    predictions_ar{test_fold} = prediction_ar;
    predictions_val{test_fold} = prediction_val;
    
    gt_ar{test_fold} = testLabels_ar{test_fold};
    gt_val{test_fold} = testLabels_val{test_fold};    
    
    [ correlations_ar{test_fold}, RMS_ar{test_fold},...
      long_correlations_ar(test_fold), long_RMS_ar(test_fold) ] = Evaluate_music_predictions(testLabels_ar{test_fold}, prediction_ar);
    
    [ correlations_val{test_fold}, RMS_val{test_fold},...
        long_correlations_val(test_fold), long_RMS_val(test_fold) ] = Evaluate_music_predictions(testLabels_val{test_fold}, prediction_val);       
            
    % Normalise the input labels to be 0-1, do the same to the
    % output ones, so that euclidean distances are the
    % comparable to other work    
    offset_ar = -1;
    scaling_ar = 2;
    
    prediction_normed_ar = (prediction_ar - offset_ar)/scaling_ar;
    test_labels_normed_ar = (gt_ar{test_fold} - offset_ar)/scaling_ar;    

    offset_val = -1;
    scaling_val = 2;
    
    prediction_normed_val = (prediction_val - offset_val)/scaling_val;
    test_labels_normed_val = (gt_val{test_fold} - offset_val)/scaling_val;

    euclidean(test_fold) = mean(sqrt((prediction_normed_ar - test_labels_normed_ar).^2 + ...
                           (prediction_normed_val - test_labels_normed_val).^2));      
                       
end

time_taken = toc;
short_correlations_ar = mean(cat(2,correlations_ar{:}));
short_correlations_val = mean(cat(2,correlations_val{:}));

short_rms_ar = mean(cat(1,RMS_ar{:}));
short_rms_val = mean(cat(1,RMS_val{:}));

save('results/GP_res', 'correlations_ar', 'RMS_ar', 'short_correlations_ar', 'long_correlations_ar', 'long_RMS_ar', 'short_rms_ar', ...
                 'correlations_val', 'RMS_val', 'short_correlations_val', 'long_correlations_val', 'long_RMS_val', 'short_rms_val',...
                 'time_taken', 'euclidean', 'predictions_ar', 'predictions_val', 'gt_ar', 'gt_val');
end

function [model] = gp_train(train_labels, train_samples, hyper)
    
    % setup the GP
    cov = {@covSEiso};
    sf = hyper.sf;
    ell = hyper.ell;                            
    
    hyp0.cov  = log([ell;sf]);
    mean = {@meanSum,{@meanLinear,@meanConst}};
    as = 1/5 * ones(size(train_samples,2),1);
    b = 1;       % m(x) = a*x+b
    
    hyp0.mean = [as;b];

    sn = hyper.sn;
                                   
    % number of conjugate gradient steps                                   
    Ncg = 50;
    
    nlZ(1) = -Inf;
    
    lik = hyper.lik{1};

    % setup the likelihood
    if strcmp(lik,'likT')
        nu = 4;
        hyp0.lik  = log([nu-1;sqrt((nu-2)/nu)*sn]);
    else
        hyp0.lik  = log(sn);
    end

    inf = hyper.inf{1};
    
    fprintf('OPT: %s/%s\n', lik, inf);
    
    % opt hypers
    hyp = minimize(hyp0,'gp', -Ncg, inf, mean, cov, lik, train_samples, train_labels); 
    
    model.hyp = hyp;
    model.mean = mean;
    model.cov = cov;
    model.lik = lik;
    model.inf = inf;

    model.train_samples = train_samples;
    model.train_labels = train_labels;
end

function [result, prediction] = gp_test(test_labels, test_samples, model)
        
    % predict
    [prediction, ~] = gp(model.hyp, model.inf, model.mean, model.cov, model.lik, model.train_samples, model.train_labels, test_samples);

    % using the average of per song RMS errors (supposed to
    % be best representation for this problem)
    prediction_reshaped = reshape(prediction, 15, numel(prediction)/15);
    test_labels_r = reshape(test_labels, 15, numel(test_labels)/15);
    rms_per_song = sqrt(mean((prediction_reshaped - test_labels_r).^2));
    result = mean(rms_per_song);  
    
end