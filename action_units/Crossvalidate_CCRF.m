function [ best_lambda_a, best_lambda_b, cv_corr_res, cv_rms_res, cv_short_corr, cv_short_rms, ...
           cv_lambda_a_vals, cv_lambda_b_vals ] = ...
            Crossvalidate_CCRF(samples_train, labels_train, samples_valid, labels_valid, similarities, ...
                threshold_x, threshold_fun, lambdas_a, lambdas_b, max_iter)
%CROSSVALIDATE_SVR Summary of this function goes here
%   Detailed explanation goes here
    
    cv_lambda_a_vals = zeros(numel(lambdas_a), numel(lambdas_b));
    cv_lambda_b_vals = zeros(numel(lambdas_a), numel(lambdas_b));
    
    cv_corr_res = zeros(numel(lambdas_a), numel(lambdas_b));
    cv_rms_res = zeros(numel(lambdas_a), numel(lambdas_b));
    cv_short_corr = zeros(numel(lambdas_a), numel(lambdas_b));
    cv_short_rms = zeros(numel(lambdas_a), numel(lambdas_b));

    input_layer_size = size(samples_train{1}, 2);
    
    n_examples = numel(samples_train);
            
    [ ~, PrecalcQ2s, PrecalcQ2sFlat, PrecalcYqDs ] = CalculateSimilarities( n_examples, samples_train, similarities, labels_train );

    for a = 1:numel(lambdas_a)

        alphas = 1 * ones(input_layer_size,1);
        betas = 1 * ones(numel(similarities), 1);

        lambda_a = 10^lambdas_a(a);

        for b = 1:numel(lambdas_b)

            lambda_b = 10^lambdas_b(b);

            cv_lambda_a_vals(a, b) = lambda_a;
            cv_lambda_b_vals(a, b) = lambda_b;

            tic
            [alphas_CCRF, betas_CCRF, scaling, ~] = CCRF_training_bfgs(n_examples, threshold_x, threshold_fun, samples_train, labels_train, labels_train, alphas, betas, lambda_a, lambda_b, similarities, PrecalcQ2s, PrecalcQ2sFlat, PrecalcYqDs, 'max_iter', max_iter);
            toc
            %--- Evaluation on test partition
            [corrs_test,rmss_test,~, ~, longCorrTest, longRMSTest] = evaluateCCRFmodel(alphas_CCRF, betas_CCRF, samples_valid, zeros(numel(samples_valid),1), labels_valid, similarities, scaling, false);

            errs = longRMSTest;
            corrs = longCorrTest;

            cv_corr_res(a,b) = cv_corr_res(a,b) + corrs;
            cv_rms_res(a,b) = cv_rms_res(a,b) + errs;
            cv_short_corr(a,b) = cv_short_corr(a,b) + mean(corrs_test);
            cv_short_rms(a,b) = cv_short_rms(a,b) + mean(rmss_test);

            fprintf('CCRF corr on validation: %.3f, rmse: %.3f \n', longCorrTest, longRMSTest);
            fprintf('alpha %.4f, beta %.4f \n', lambda_a, lambda_b);
        end
    end
    
    %% Finding the best values of regularisations and the number of layers
    [val,~] = min(min(cv_rms_res));
    [a, b] = ind2sub(size(cv_rms_res), find(cv_rms_res == val));
    
    best_lambda_a = cv_lambda_a_vals(a(1), b(1));
    best_lambda_b = cv_lambda_b_vals(a(1), b(1));
            
end