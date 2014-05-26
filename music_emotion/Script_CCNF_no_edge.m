clear

addpath('../CCNF/lib/');

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
best_lambdas_th_ar = zeros(num_test_folds,1);
best_num_layers_ar = zeros(num_test_folds,1);

best_lambdas_a_val = zeros(num_test_folds,1);
best_lambdas_b_val = zeros(num_test_folds,1);
best_lambdas_th_val = zeros(num_test_folds,1);
best_num_layers_val = zeros(num_test_folds,1);

[trainSamples, testSamples, ...
trainLabels_ar, trainLabels_ar_normed, testLabels_ar,...
trainLabels_val, trainLabels_val_normed, testLabels_val,...
offsets_val, offsets_ar, scalings_val, scalings_ar, num_features] = Prepare_CCNF_music_data(num_test_folds);

% The actual cross-validation results, as they would be useful for just
% rerunning results, without the need to crossvalidate
cv_corrs_long_ar = cell(num_test_folds, 1);
cv_corrs_long_val = cell(num_test_folds, 1);
cv_rmss_long_ar = cell(num_test_folds, 1);
cv_rmss_long_val = cell(num_test_folds, 1);

cv_corrs_ar = cell(num_test_folds, 1);
cv_corrs_val = cell(num_test_folds, 1);
cv_rmss_ar = cell(num_test_folds, 1);
cv_rmss_val = cell(num_test_folds, 1);


%% Preparing the similarity functions
% number of similarity functions
nGaussians = 0;
nNeighbor = 0;
   
% this will create a family of exponential decays with different sigmas
similarities = {};
    
range = 1:num_features; % all features used for the exponential similarity

for i=1:nNeighbor
    neigh = i;
    neighFn = @(x) similarityNeighbor(x, neigh, range);
    similarities = cat(1, similarities, {neighFn});
end

%% Some optimisation parameters
threshold_x = 1e-8;
threshold_fun = 1e-4;
input_layer_size  = num_features;
max_iter = 300;

%% hyper-parameters to cross-validate
lambdas_a = [0,1,2];
lambdas_b = [-3,-2,-1,0];
lambdas_th = [-3,-2,-1,0];
neural_layers = [10,20,30];

tic
for test_fold = 1:num_test_folds

    num_cv_folds = 2;

    reinit = true;

    % do the arousal bit first
    [ best_lambda_a_ar, best_lambda_b_ar, best_lambda_th_ar, best_num_layer_ar,...
      cv_corr_res, cv_rms_res, cv_short_corr, cv_short_rms, ~, ~, ~, ~ ] = ...
               Crossvalidate_CCNF(trainSamples{test_fold}, trainLabels_ar_normed{test_fold}, similarities, ...
                num_cv_folds, threshold_x, threshold_fun, lambdas_a, lambdas_b, lambdas_th, neural_layers, reinit, 'lbfgs', 'max_iter', max_iter);

    % save the information about the current fold        
    cv_corrs_long_ar{test_fold} = cv_corr_res;
    cv_rmss_long_ar{test_fold} = cv_rms_res;    
    cv_corrs_ar{test_fold} = cv_short_corr;
    cv_rmss_ar{test_fold} = cv_short_rms;

    best_lambdas_a_ar(test_fold) = best_lambda_a_ar;
    best_lambdas_b_ar(test_fold) = best_lambda_b_ar;
    best_lambdas_th_ar(test_fold) = best_lambda_th_ar;
    best_num_layers_ar(test_fold) = best_num_layer_ar;

    [ best_lambda_a_val, best_lambda_b_val, best_lambda_th_val, best_num_layer_val,...
      cv_corr_res_val, cv_rms_res_val, cv_short_corr_val, cv_short_rms_val, ...
      cv_lambda_a_vals, cv_lambda_b_vals, cv_lambda_th_vals, cv_num_layers ] = ...
               Crossvalidate_CCNF(trainSamples{test_fold}, trainLabels_val_normed{test_fold}, similarities, ...
                num_cv_folds, threshold_x, threshold_fun, lambdas_a, lambdas_b, lambdas_th, neural_layers, reinit, 'lbfgs', 'max_iter', max_iter);

    % save the information about the current fold        
    cv_corrs_long_ar{test_fold} = cv_corr_res_val;
    cv_rmss_long_ar{test_fold} = cv_rms_res_val;    
    cv_corrs_ar{test_fold} = cv_short_corr_val;
    cv_rmss_ar{test_fold} = cv_short_rms_val;

    best_lambdas_a_val(test_fold) = best_lambda_a_val;
    best_lambdas_b_val(test_fold) = best_lambda_b_val;
    best_lambdas_th_val(test_fold) = best_lambda_th_val;
    best_num_layers_val(test_fold) = best_num_layer_val;    

    %%
    numRedo = 20;
    likelihoods_ar = zeros(numRedo, 1);
    alphas_ar = cell(numRedo,1);
    betas_ar = cell(numRedo,1);
    thetas_ar = cell(numRedo,1);

    likelihoods_val = zeros(numRedo, 1);
    alphas_val = cell(numRedo,1);
    betas_val = cell(numRedo,1);
    thetas_val = cell(numRedo,1);
    
    %% Keep redoing because of local optimum
    for i=1:numRedo
        
        % arousal
        initial_Theta = randInitializeWeights(input_layer_size, best_num_layers_ar(test_fold));
        alphas = 1 * ones(best_num_layers_ar(test_fold),1);
        betas = 1 * ones(numel(similarities), 1);
        
        [alphas_ar{i}, betas_ar{i}, thetas_ar{i}, likelihoods_ar(i)] = CCNF_training_bfgs(threshold_x, threshold_fun, trainSamples{test_fold}, trainLabels_ar_normed{test_fold}, alphas, betas, initial_Theta, best_lambda_a_ar, best_lambda_b_ar, best_lambda_th_ar, similarities,[],'const',true, 'reinit', true, 'lbfgs', 'max_iter', max_iter);
        
        % valence
        initial_Theta = randInitializeWeights(input_layer_size, best_num_layers_val(test_fold));
        alphas = 1 * ones(best_num_layers_val(test_fold),1);
        betas = 1 * ones(numel(similarities), 1);
        
        % training valence
        [alphas_val{i}, betas_val{i}, thetas_val{i}, likelihoods_val(i)] = CCNF_training_bfgs(threshold_x, threshold_fun, trainSamples{test_fold}, trainLabels_val_normed{test_fold}, alphas, betas, initial_Theta, best_lambda_a_val, best_lambda_b_val, best_lambda_th_val, similarities,[],'const',true, 'reinit', true, 'lbfgs', 'max_iter', max_iter);        
    end
    [val_ar,ind] = max(likelihoods_ar);
    fprintf('-------------------------------------------------------\n');
    
    [correlations_ar{test_fold}, RMS_ar{test_fold},~, ~,...
        long_correlations_ar(test_fold), long_RMS_ar(test_fold),...
        pred_ar, gts_ar ] = evaluate_CCNF_model(alphas_ar{ind}, betas_ar{ind}, thetas_ar{ind}, testSamples{test_fold}, testLabels_ar{test_fold}, similarities, [], offsets_ar{test_fold}, scalings_ar{test_fold}, false);
    fprintf('CCNF corr on test arousal: %.3f, rmse: %.3f (%d, %d, %f, %d) \n', long_correlations_ar(test_fold), long_RMS_ar(test_fold), best_lambda_a_ar, best_lambda_b_ar, best_lambda_th_ar, best_num_layers_ar(test_fold));
           
    [val_val,ind] = max(likelihoods_val);
    
    [ correlations_val{test_fold}, RMS_val{test_fold},~,~,...
        long_correlations_val(test_fold), long_RMS_val(test_fold),...
        pred_val, gts_val ] = evaluate_CCNF_model(alphas_val{ind}, betas_val{ind}, thetas_val{ind}, testSamples{test_fold}, testLabels_val{test_fold}, similarities, [], offsets_val{test_fold}, scalings_val{test_fold}, false);
    fprintf('CCNF corr on test valence: %.3f, rmse: %.3f (%d, %d, %f, %d)\n', long_correlations_val(test_fold), long_RMS_val(test_fold), best_lambda_a_val, best_lambda_b_val, best_lambda_th_val, best_num_layers_val(test_fold));
    
    predictions_ar{test_fold} = pred_ar;
    gt_ar{test_fold} = gts_ar;     
    
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

%%
save('results/CCNF_no_edge', 'best_lambdas_a_ar', 'best_lambdas_b_ar', 'best_lambdas_th_ar', 'best_num_layers_ar', ...
                                                 'best_lambdas_a_val', 'best_lambdas_b_val', 'best_lambdas_th_val', 'best_num_layers_val', ...
                                 'correlations_ar', 'RMS_ar', 'short_correlations_ar', 'long_correlations_ar', 'long_RMS_ar', 'short_rms_ar',...
                                 'correlations_val', 'RMS_val', 'short_correlations_val', 'long_correlations_val', 'long_RMS_val', 'short_rms_val',...
                                 'time_taken', 'euclidean', 'predictions_ar', 'predictions_val', 'gt_ar', 'gt_val',...
                                'cv_lambda_a_vals', 'cv_lambda_b_vals', 'cv_lambda_th_vals', 'cv_num_layers', 'cv_corrs_ar', 'cv_corrs_val', ... % crossvalidations stuff
                                'cv_rmss_ar', 'cv_rmss_val');   