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

best_cs_ar = zeros(num_test_folds,1);
best_ps_ar = zeros(num_test_folds,1);

best_cs_val = zeros(num_test_folds,1);
best_ps_val = zeros(num_test_folds,1);

euclidean = zeros(num_test_folds, 1);

predictions_ar = cell(num_test_folds, 1);
predictions_val = cell(num_test_folds, 1);

gt_ar = cell(num_test_folds, 1);
gt_val = cell(num_test_folds, 1);

[trainSamples, testSamples, trainLabels_ar, trainLabels_ar_normed,...
    testLabels_ar, trainLabels_val, trainLabels_val_normed, testLabels_val,...
    offsets_val, offsets_ar, scalings_val, scalings_ar, num_features] = Prepare_SVR_music_data(num_test_folds, true);

%%
numCVFolds = 2;
sequence_length = 15;

[ indsCV_train, indsCV_test ] = GetFolds(size(trainSamples{1},1)/sequence_length, numCVFolds);

% Hyper-parameters
cs = [-6:3];
ps = [-8:-1];

tic
for test_fold = 1:num_test_folds

    cvCvals = zeros(numel(cs), numel(ps));
    cvPvals = zeros(numel(cs), numel(ps));

    cvCorrRes_ar = zeros(numel(cs), numel(ps));
    cvRmsRes_ar = zeros(numel(cs), numel(ps));

    cvCorrRes_val = zeros(numel(cs), numel(ps));
    cvRmsRes_val = zeros(numel(cs), numel(ps));

    % We are using epsilon SVR here

    for i = 1:numCVFolds

        % as we don't want to break up songs make sure the train and test
        % use the same songs
        train_indices = indsCV_train(:,i) * ones(1, sequence_length);
        train_indices = train_indices';
        train_indices = logical(train_indices(:));
        
        test_indices = indsCV_test(:,i) * ones(1, sequence_length);
        test_indices = test_indices';
        test_indices = logical(test_indices(:));
        
        trainLabelsCV_ar = trainLabels_ar{test_fold}(train_indices,:);
        trainLabelsCV_val = trainLabels_val{test_fold}(train_indices,:);

        trainSamplesCV = trainSamples{test_fold}(train_indices,:);

        testLabelsCV_ar = trainLabels_ar{test_fold}(test_indices,:);
        testLabelsCV_val = trainLabels_val{test_fold}(test_indices,:);

        testSamplesCV = trainSamples{test_fold}(test_indices,:);

        % Crossvalidate the C and epsilon values
        for c = 1:numel(cs)

            cCurr = 10^cs(c);

            for p = 1:numel(ps)

                pCurr = 2^ps(p);
                                
                cvCvals(c,p) = cCurr;
                cvPvals(c,p) = pCurr;
                
                comm = sprintf('-s 3 -t 0 -p %f -c %f -q', pCurr, cCurr);
                model_ar = svmtrain(trainLabelsCV_ar, trainSamplesCV, comm);
                model_val = svmtrain(trainLabelsCV_val, trainSamplesCV, comm);
                
                % calc error
                prediction_ar = svmpredict(testLabelsCV_ar, testSamplesCV, model_ar);

                % Using the long correlation
                corrs_ar = corr(testLabelsCV_ar, prediction_ar)^2;
                                
                % using the average of per song RMS errors (supposed to
                % be bes representation)
                prediction = reshape(prediction_ar, 15, numel(prediction_ar)/15);
                test_labels = reshape(testLabelsCV_ar, 15, numel(testLabelsCV_ar)/15);
                  
                rms_per_song = sqrt(mean((prediction - test_labels).^2));
                errs_ar = mean(rms_per_song);                


                cvCorrRes_ar(c,p) = cvCorrRes_ar(c,p) + corrs_ar;
                cvRmsRes_ar(c,p) = cvRmsRes_ar(c,p) + errs_ar;
                
                prediction_val = svmpredict(testLabelsCV_val, testSamplesCV, model_val);
                
                % Using the long correlation
                corrs_val = corr(testLabelsCV_val, prediction_val)^2;
                                
                % using the average of per song RMS errors (supposed to
                % be bes representation)
                prediction = reshape(prediction_val, 15, numel(prediction_val)/15);
                test_labels = reshape(testLabelsCV_val, 15, numel(testLabelsCV_val)/15);
                  
                rms_per_song = sqrt(mean((prediction - test_labels).^2));
                errs_val = mean(rms_per_song);   
                
                cvCorrRes_val(c,p) = cvCorrRes_val(c,p) + corrs_val;
                cvRmsRes_val(c,p) = cvRmsRes_val(c,p) + errs_val;
            end
        end

    end

    cvCorrRes_ar = cvCorrRes_ar / numCVFolds;
    cvCorrRes_val = cvCorrRes_val / numCVFolds;
    cvRmsRes_ar = cvRmsRes_ar / numCVFolds;
    cvRmsRes_val = cvRmsRes_val / numCVFolds;

    %% Finding the best values of c and p and g

    [val_ar,~] = min(min(cvRmsRes_ar));
    [a, b] = ind2sub(size(cvRmsRes_ar), find(cvRmsRes_ar == val_ar));

    best_c_ar = cvCvals(a(1), b(1));
    best_p_ar = cvPvals(a(1), b(1));

    [val_val,~] = min(min(cvRmsRes_val));
    [a, b] = ind2sub(size(cvRmsRes_val), find(cvRmsRes_val == val_val));

    best_c_val = cvCvals(a(1),b(1));
    best_p_val = cvPvals(a(1),b(1));

    %% The actual testing results

    comm = sprintf('-s 3 -t 0 -p %f -c %f -q', best_p_ar, best_c_ar);
    model_ar = svmtrain(trainLabels_ar{test_fold}, trainSamples{test_fold}, comm);

    comm = sprintf('-s 3 -t 0 -p %f -c %f -q', best_p_val, best_c_val);
    model_val = svmtrain(trainLabels_val{test_fold}, trainSamples{test_fold}, comm);
% 
%     prediction_ar = svmpredict(trainLabels_ar{test_fold}, trainSamples{test_fold}, model_ar);
%     prediction_val = svmpredict(trainLabels_val{test_fold}, trainSamples{test_fold}, model_val);

    prediction_ar = svmpredict(testLabels_ar{test_fold}, testSamples{test_fold}, model_ar);
    prediction_val = svmpredict(testLabels_val{test_fold}, testSamples{test_fold}, model_val);    

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
    test_labels_normed_ar = (testLabels_ar{test_fold} - offset_ar)/scaling_ar;    

    offset_val = -1;
    scaling_val = 2;
    
    prediction_normed_val = (prediction_val - offset_val)/scaling_val;
    test_labels_normed_val = (testLabels_val{test_fold} - offset_val)/scaling_val;

    euclidean(test_fold) = mean(sqrt((prediction_normed_ar - test_labels_normed_ar).^2 + ...
                           (prediction_normed_val - test_labels_normed_val).^2));    

    % remember the cross-validation results
    best_cs_ar(test_fold) = best_c_ar;
    best_ps_ar(test_fold) = best_p_ar;

    best_cs_val(test_fold) = best_c_val;
    best_ps_val(test_fold) = best_p_val;

    
end
time_taken = toc;

short_correlations_ar = mean(cat(2,correlations_ar{:}));
short_correlations_val = mean(cat(2,correlations_val{:}));

short_rms_ar = mean(cat(1,RMS_ar{:}));
short_rms_val = mean(cat(1,RMS_val{:}));

save('results/SVR_linear', 'best_cs_ar', 'best_ps_ar', ...
                                 'best_cs_val', 'best_ps_val', ...
                                 'correlations_ar', 'RMS_ar', 'short_correlations_ar', 'long_correlations_ar', 'long_RMS_ar', 'short_rms_ar',...
                                 'correlations_val', 'RMS_val', 'short_correlations_val', 'long_correlations_val', 'long_RMS_val', 'short_rms_val',...
                                 'time_taken', 'euclidean', 'predictions_ar', 'predictions_val', 'gt_ar', 'gt_val');
