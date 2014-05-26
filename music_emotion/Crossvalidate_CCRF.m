function [ best_lambda_a, best_lambda_b, ...
           cv_corr_res, cv_rms_res, cv_short_corr, cv_short_rms, ...
           cv_lambda_a_vals, cv_lambda_b_vals ] = ...
            Crossvalidate_CCRF(samples, labels, labels_unnormed, similarities, ...
                num_cv_folds, threshold_x, threshold_fun, lambdas_a, lambdas_b)
%CROSSVALIDATE_SVR Summary of this function goes here
%   Detailed explanation goes here
    
    cv_lambda_a_vals = zeros(numel(lambdas_a), numel(lambdas_b));
    cv_lambda_b_vals = zeros(numel(lambdas_a), numel(lambdas_b));
    
    cv_corr_res = zeros(numel(lambdas_a), numel(lambdas_b));
    cv_rms_res = zeros(numel(lambdas_a), numel(lambdas_b));
    cv_short_corr = zeros(numel(lambdas_a), numel(lambdas_b));
    cv_short_rms = zeros(numel(lambdas_a), numel(lambdas_b));

    [ inds_cv_train, inds_cv_test ] = GetFolds(size(samples,1), num_cv_folds);

    % CCNF crossvalidation here
    for i = 1:num_cv_folds
               
        train_labels = labels(inds_cv_train(:,i),:);
        train_labels_unnormed = labels_unnormed(inds_cv_train(:,i),:);
        train_samples = samples(inds_cv_train(:,i),:);
                
        input_layer_size = size(train_samples{1}, 2);
                
        test_labels = labels(inds_cv_test(:,i),:);
        test_samples = samples(inds_cv_test(:,i),:);
        test_labels_unnormed = labels_unnormed(inds_cv_test(:,i),:);
        
        for a = 1:numel(lambdas_a)
            
            alphas = 1 * ones(input_layer_size,1);
            betas = 1 * ones(numel(similarities), 1);
            
            lambda_a = 10^lambdas_a(a);
            
            for b = 1:numel(lambdas_b)
                
                lambda_b = 10^lambdas_b(b);
                
                cv_lambda_a_vals(a, b) = lambda_a;
                cv_lambda_b_vals(a, b) = lambda_b;
                
                % Training
                n_examples = numel(train_samples);
                [alphas_CCRF, betas_CCRF, scaling, ~] = CCRF_training_bfgs(n_examples, threshold_x, threshold_fun, train_samples, train_labels, train_labels_unnormed, cell(numel(train_labels),1), alphas, betas, lambda_a, lambda_b, similarities, false);

                %--- Evaluation on test partition
                [corrs_test,rmss_test,~, mean_rms, longCorrTest, longRMSTest] = evaluateCCRFmodel(alphas_CCRF, betas_CCRF, test_samples, zeros(numel(test_samples),1), test_labels_unnormed, cell(numel(train_labels),1), false, similarities, scaling, false);
                
                errs = mean_rms;
                corrs = longCorrTest;
                
                cv_corr_res(a,b) = cv_corr_res(a,b) + corrs;
                cv_rms_res(a,b) = cv_rms_res(a,b) + errs;
                cv_short_corr(a,b) = cv_short_corr(a,b) + mean(corrs_test);
                cv_short_rms(a,b) = cv_short_rms(a,b) + mean(rmss_test);
                
                fprintf('CCNF corr on test arousal: %.3f, rmse: %.3f ', longCorrTest, longRMSTest);
                fprintf('alpha %.4f, beta %.4f \n', lambda_a, lambda_b);
            end
        end
    end
    
    cv_corr_res = cv_corr_res / num_cv_folds;
    cv_rms_res = cv_rms_res / num_cv_folds;    
    cv_short_corr = cv_short_corr / num_cv_folds;
    cv_short_rms = cv_short_rms / num_cv_folds;
    
    %% Finding the best values of regularisations and the number of layers
    [val,~] = min(min(cv_rms_res));
    [a, b] = ind2sub(size(cv_rms_res), find(cv_rms_res == val));
    
    best_lambda_a = cv_lambda_a_vals(a(1), b(1));
    best_lambda_b = cv_lambda_b_vals(a(1), b(1));
        
    
end