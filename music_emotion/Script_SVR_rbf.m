clear

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

best_cs_ar = zeros(num_test_folds,1);
best_ps_ar = zeros(num_test_folds,1);
best_gs_ar = zeros(num_test_folds,1);

best_cs_val = zeros(num_test_folds,1);
best_ps_val = zeros(num_test_folds,1);
best_gs_val = zeros(num_test_folds,1);

% load the training and testing data
[trainSamples, testSamples, trainLabels_ar, trainLabels_ar_normed,...
    testLabels_ar, trainLabels_val, trainLabels_val_normed, testLabels_val,...
    offsets_val, offsets_ar, scalings_val, scalings_ar, num_features] = Prepare_CCNF_music_data(num_test_folds);

%%
num_cv_folds = 2;
sequence_length = 15;

tic
for test_fold = 1:num_test_folds

    % Cross-validate here
    [best_c_ar, best_p_ar, best_g_ar, corrs_ar, rmss_ar,...
        ~,~,~] = Crossvalidate_SVR_rbf(trainSamples{test_fold}, trainLabels_ar{test_fold}, num_cv_folds, true);
    [best_c_val, best_p_val, best_g_val, corrs_val, rmss_val,...
        cv_c_vals,cv_p_vals,cv_g_vals] = Crossvalidate_SVR_rbf(trainSamples{test_fold}, trainLabels_val{test_fold}, num_cv_folds, true);
   
    %% The actual testing results

    comm = sprintf('-s 3 -t 2 -p %f -c %f -g %f -q', best_p_ar, best_c_ar, best_g_ar);
    model_ar = svmtrain(cell2mat(trainLabels_ar{test_fold}), cell2mat(trainSamples{test_fold}), comm);

    comm = sprintf('-s 3 -t 2 -p %f -c %f -g %f -q', best_p_val, best_c_val, best_g_val);
    model_val = svmtrain(cell2mat(trainLabels_val{test_fold}), cell2mat(trainSamples{test_fold}), comm);
% 
%     prediction_ar = svmpredict(trainLabels_ar{test_fold}, trainSamples{test_fold}, model_ar);
%     prediction_val = svmpredict(trainLabels_val{test_fold}, trainSamples{test_fold}, model_val);

    prediction_ar = svmpredict(cell2mat(testLabels_ar{test_fold}), cell2mat(testSamples{test_fold}), model_ar);
    prediction_val = svmpredict(cell2mat(testLabels_val{test_fold}), cell2mat(testSamples{test_fold}), model_val);    

    predictions_ar{test_fold} = prediction_ar;
    predictions_val{test_fold} = prediction_val;
    
    gt_ar{test_fold} = cell2mat(testLabels_ar{test_fold});
    gt_val{test_fold} = cell2mat(testLabels_val{test_fold});    
    
    [ correlations_ar{test_fold}, RMS_ar{test_fold},...
      long_correlations_ar(test_fold), long_RMS_ar(test_fold) ] = Evaluate_music_predictions(cell2mat(testLabels_ar{test_fold}), prediction_ar);
    
    [ correlations_val{test_fold}, RMS_val{test_fold},...
        long_correlations_val(test_fold), long_RMS_val(test_fold) ] = Evaluate_music_predictions(cell2mat(testLabels_val{test_fold}), prediction_val);       
            
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
    
    best_cs_ar(test_fold) = best_c_ar;
    best_ps_ar(test_fold) = best_p_ar;
    best_gs_ar(test_fold) = best_g_ar;

    best_cs_val(test_fold) = best_c_val;
    best_ps_val(test_fold) = best_p_val;
    best_gs_val(test_fold) = best_g_val;

    
end
time_taken = toc;
short_correlations_ar = mean(cat(2,correlations_ar{:}));
short_correlations_val = mean(cat(2,correlations_val{:}));

short_rms_ar = mean(cat(1,RMS_ar{:}));
short_rms_val = mean(cat(1,RMS_val{:}));

save('results/SVR_rbf', 'best_cs_ar', 'best_ps_ar', ...
                                 'best_cs_val', 'best_ps_val', ...
                                 'correlations_ar', 'RMS_ar', 'short_correlations_ar', 'long_correlations_ar', 'long_RMS_ar', 'short_rms_ar', ...
                                 'correlations_val', 'RMS_val', 'short_correlations_val', 'long_correlations_val', 'long_RMS_val', 'short_rms_val',...
                                 'time_taken', 'euclidean', 'predictions_ar', 'predictions_val', 'gt_ar', 'gt_val');

