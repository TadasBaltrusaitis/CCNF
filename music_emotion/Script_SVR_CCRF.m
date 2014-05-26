clear

addpath('../CCRF/lib/');

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

best_lambdas_a_ar = zeros(num_test_folds,1);
best_lambdas_b_ar = zeros(num_test_folds,1);

best_lambdas_a_val = zeros(num_test_folds,1);
best_lambdas_b_val = zeros(num_test_folds,1);

[trainSamples, testSamples, ...
trainLabels_ar, trainLabels_ar_normed, testLabels_ar,...
trainLabels_val, trainLabels_val_normed, testLabels_val,...
offsets_val, offsets_ar, scalings_val, scalings_ar, num_features] = Prepare_CCNF_music_data(num_test_folds);

%% Preparing the similarity functions to be used by CCRF
% number of similarity functions
nNeighbor = 1;

% this will create a family of exponential decays with different sigmas
similarities = {};

range = 1:num_features; % all features used for the exponential similarity

for i=1:nNeighbor
    neigh = i;
    neighFn = @(x) similarityNeighbor(x, neigh, range);
    similarities = cat(1, similarities, {neighFn});
end

%%
num_cv_folds = 2;
tic
for test_fold = 1:num_test_folds
    
    % split the train and test into two parts (SVR and CCRF)
    cv_samples_svr = trainSamples{test_fold}(1:end/2);
    cv_samples_ccrf = trainSamples{test_fold}(end/2+1:end);

    cv_labels_ar_svr = trainLabels_ar{test_fold}(1:end/2);
    cv_labels_ar_ccrf = trainLabels_ar{test_fold}(end/2+1:end);

    cv_labels_val_svr = trainLabels_val{test_fold}(1:end/2);
    cv_labels_val_ccrf = trainLabels_val{test_fold}(end/2+1:end);
    
    %% first train the svr on per frame basis
    
    % cross-validate it
    [best_c_ar, best_p_ar, best_g_ar, corrs_ar, rmss_ar,...
        ~,~,~] = Crossvalidate_SVR_rbf(cv_samples_svr, cv_labels_ar_svr, num_cv_folds, true);
    [best_c_val, best_p_val, best_g_val, corrs_val, rmss_val,...
        cv_c_vals,cv_p_vals,cv_g_vals] = Crossvalidate_SVR_rbf(cv_samples_svr, cv_labels_val_svr, num_cv_folds, true);    
        
    %% Now that the SVR hyper-parameters are found we can train an SVR model and apply it on the CCRF partition
    comm = sprintf('-s 3 -t 2 -p %f -c %f -g %f -h 0 -q', best_p_ar, best_c_ar, best_g_ar);
    % TODO train just on itself or on the whole train set
    model_ar = svmtrain(cell2mat(trainLabels_ar{test_fold}), cell2mat(trainSamples{test_fold}), comm);

    comm = sprintf('-s 3 -t 2 -p %f -c %f -g %f -h 0 -q', best_p_val, best_c_val, best_g_val);
    model_val = svmtrain(cell2mat(trainLabels_val{test_fold}), cell2mat(trainSamples{test_fold}), comm);
    
    %% use SVR to predict the arousal and valence on the CCRF sample
    prediction_ar = svmpredict(cell2mat(cv_labels_ar_ccrf), cell2mat(cv_samples_ccrf), model_ar);
    prediction_val = svmpredict(cell2mat(cv_labels_val_ccrf), cell2mat(cv_samples_ccrf), model_val);    
      
    cv_samples_ccrf_ar = mat2cell(prediction_ar, 15*ones(numel(cv_samples_ccrf),1), 1);
    cv_samples_ccrf_val = mat2cell(prediction_val, 15*ones(numel(cv_samples_ccrf),1), 1);

    %% Some parameters
    threshold_x = 1e-10;
    threshold_fun = 1e-6;
    lambdas_a = -3:3;
    lambdas_b = -3:3;
    % do the arousal bit first
    
    [ best_lambda_a_ar, best_lambda_b_ar, cv_corr_res_ar, cv_rms_res_ar, cv_short_corr_ar, cv_short_rms_ar, ~, ~] = ...
               Crossvalidate_CCRF(cv_samples_ccrf_ar, cv_labels_ar_ccrf, cv_labels_ar_ccrf, similarities, ...
                num_cv_folds, threshold_x, threshold_fun, lambdas_a, lambdas_b);

    % save the information about the current fold        
    best_lambdas_a_ar(test_fold) = best_lambda_a_ar;
    best_lambdas_b_ar(test_fold) = best_lambda_b_ar;

    [ best_lambda_a_val, best_lambda_b_val, cv_corr_res_val, cv_rms_res_val, cv_short_corr_val, cv_short_rms_val, ~, ~] = ...
               Crossvalidate_CCRF(cv_samples_ccrf_val, cv_labels_val_ccrf, cv_labels_val_ccrf, similarities, ...
                num_cv_folds, threshold_x, threshold_fun, lambdas_a, lambdas_b);

    % save the information about the current fold        
    best_lambdas_a_val(test_fold) = best_lambda_a_val;
    best_lambdas_b_val(test_fold) = best_lambda_b_val;

    %% Finally do the testing
    
    alphas = 1 * ones(1,1);
    betas = 1 * ones(numel(similarities), 1);

    % train on the whole set now
    n_examples = numel(cv_samples_ccrf_ar);
    [alphas_ar, betas_ar, scaling, ~] = ...
        CCRF_training_bfgs(n_examples, threshold_x, threshold_fun, cv_samples_ccrf_ar, cv_labels_ar_ccrf, cv_labels_ar_ccrf, cell(numel(cv_labels_ar_ccrf),1), alphas, betas, best_lambda_a_ar, best_lambda_b_ar, similarities, false);
    scaling = 1;
    % Do svr prediction on the test set
    prediction_ar = svmpredict(cell2mat(testLabels_ar{test_fold}), cell2mat(testSamples{test_fold}), model_ar);
    % Convert them to cell represenation
    prediction_ar = mat2cell(prediction_ar, 15*ones(numel(testLabels_ar{test_fold}),1), 1);
    
    % calculate the offsets for the prediction
    x_offsets_ar = zeros(size(testLabels_ar{test_fold}));
    
    % Evaluate on the test set
    [correlations_ar{test_fold}, RMS_ar{test_fold},~, ~,long_correlations_ar(test_fold), long_RMS_ar(test_fold), pred_ar, gts_ar ] = ...
            evaluateCCRFmodel(alphas_ar, betas_ar, prediction_ar, x_offsets_ar, testLabels_ar{test_fold}, cell(numel(testLabels_ar{test_fold}),1), false, similarities, scaling, false);
    fprintf('CCRF corr on test arousal: %.3f, rmse: %.3f \n', long_correlations_ar(test_fold), long_RMS_ar(test_fold));

    predictions_ar{test_fold} = pred_ar;
    gt_ar{test_fold} = gts_ar; 
                    
    [alphas_val, betas_val, scaling, ~] = ...
        CCRF_training_bfgs(n_examples, threshold_x, threshold_fun, cv_samples_ccrf_val, cv_labels_val_ccrf, cv_labels_val_ccrf, cell(numel(cv_labels_val_ccrf),1), alphas, betas, best_lambda_a_val, best_lambda_b_val, similarities, false);
    scaling = 1;
    % Do svr prediction on the test set
    prediction_val = svmpredict(cell2mat(testLabels_val{test_fold}), cell2mat(testSamples{test_fold}), model_val);
    % Convert them to cell represenation
    prediction_val = mat2cell(prediction_val, 15*ones(numel(testLabels_val{test_fold}),1), 1);
        
    % calculate the offsets for the prediction
    x_offsets_val = zeros(size(testLabels_val{test_fold}));
    
    % Evaluate on the test set
    [correlations_val{test_fold}, RMS_val{test_fold},~, ~,long_correlations_val(test_fold), long_RMS_val(test_fold), pred_val, gts_val ] = ...
            evaluateCCRFmodel(alphas_val, betas_val, prediction_val, x_offsets_val, testLabels_val{test_fold}, cell(numel(testLabels_val{test_fold}),1), false, similarities, scaling, false);

    fprintf('CCRF corr on test valence: %.3f, rmse: %.3f \n', long_correlations_val(test_fold), long_RMS_val(test_fold));

    predictions_val{test_fold} = pred_val;
    gt_val{test_fold} = gts_val;
       
    % Normalise the input labels to be 0-1, do the same to the
    % output ones, so that euclidean distances are the
    % comparable to other work    
    offset_ar = -1;
    scaling_ar = 2;
    
    prediction_normed_ar = (pred_ar - offset_ar)/scaling_ar;
    test_labels_normed_ar = (gts_ar - offset_ar)/scaling_ar;    

    offset_val = -1;
    scaling_val = 2;
    
    prediction_normed_val = (pred_val - offset_val)/scaling_val;
    test_labels_normed_val = (gts_val - offset_val)/scaling_val;

    euclidean(test_fold) = mean(sqrt((prediction_normed_ar - test_labels_normed_ar).^2 + ...
                           (prediction_normed_val - test_labels_normed_val).^2));        

    fprintf('--------------------------------- %d done ---------------------\n', test_fold);
    

end
time_taken = toc;

short_correlations_ar = mean(cat(2,correlations_ar{:}));
short_correlations_val = mean(cat(2,correlations_val{:}));

short_rms_ar = mean(cat(1,RMS_ar{:}));
short_rms_val = mean(cat(1,RMS_val{:}));

save('results/CCRF', 'best_lambdas_a_ar', 'best_lambdas_b_ar', 'best_lambdas_a_val', 'best_lambdas_b_val',...
                                 'correlations_ar', 'RMS_ar', 'short_correlations_ar', 'long_correlations_ar', 'long_RMS_ar', 'short_rms_ar', ...
                                 'correlations_val', 'RMS_val', 'short_correlations_val', 'long_correlations_val', 'long_RMS_val', 'short_rms_val', ...
                                 'time_taken', 'euclidean', 'predictions_ar', 'predictions_val', 'gt_ar', 'gt_val');                                 