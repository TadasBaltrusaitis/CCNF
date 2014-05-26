function [ best_lambda_a, best_lambda_b, best_lambda_th, best_num_layer,...
           cv_corr_res, cv_rms_res, cv_short_corr, cv_short_rms, ...
           cv_lambda_a_vals, cv_lambda_b_vals, cv_lambda_th_vals, cv_num_layers ] = ...
            Crossvalidate_CCNF(samples, labels, similarities, ...
                num_cv_folds, threshold_x, threshold_fun, lambdas_a, lambdas_b, lambdas_th, neural_layers, reinit, varargin)
%CROSSVALIDATE_SVR Summary of this function goes here
%   Detailed explanation goes here
    
    cv_lambda_a_vals = zeros(numel(neural_layers), numel(lambdas_a), numel(lambdas_b), numel(lambdas_th));
    cv_lambda_b_vals = zeros(numel(neural_layers), numel(lambdas_a), numel(lambdas_b), numel(lambdas_th));
    cv_lambda_th_vals = zeros(numel(neural_layers), numel(lambdas_a), numel(lambdas_b), numel(lambdas_th));
    cv_num_layers = zeros(numel(neural_layers), numel(lambdas_a), numel(lambdas_b), numel(lambdas_th));
    
    cv_corr_res = zeros(numel(neural_layers), numel(lambdas_a), numel(lambdas_b), numel(lambdas_th));
    cv_rms_res = zeros(numel(neural_layers), numel(lambdas_a), numel(lambdas_b), numel(lambdas_th));
    cv_short_corr = zeros(numel(neural_layers), numel(lambdas_a), numel(lambdas_b), numel(lambdas_th));
    cv_short_rms = zeros(numel(neural_layers), numel(lambdas_a), numel(lambdas_b), numel(lambdas_th));

    [ inds_cv_train, inds_cv_test ] = GetFolds(size(samples,1), num_cv_folds);

    % CCNF crossvalidation here
    for i = 1:num_cv_folds
               
        train_labels = labels(inds_cv_train(:,i),:);
        
        train_samples = samples(inds_cv_train(:,i),:);
                
        input_layer_size = size(train_samples{1}, 2);
                
        test_labels = labels(inds_cv_test(:,i),:);
        test_samples = samples(inds_cv_test(:,i),:);
        
        for n=1:numel(neural_layers)
            for a = 1:numel(lambdas_a)
                
                num_alphas = neural_layers(n);
                
                alphas = 1 * ones(num_alphas,1);
                betas = 1 * ones(numel(similarities), 1);
                
                lambda_a = 10^lambdas_a(a);
                
                for b = 1:numel(lambdas_b)
                    
                    lambda_b = 10^lambdas_b(b);
                    
                    for th = 1:numel(lambdas_th)
                        
                        lambda_th  = 10^lambdas_th(th);
                        
                        cv_lambda_a_vals(n, a, b, th) = lambda_a;
                        cv_lambda_b_vals(n, a, b, th) = lambda_b;
                        cv_lambda_th_vals(n, a, b, th) = lambda_th;
                        cv_num_layers(n, a, b, th) = num_alphas;

                        num_redo = 1;
                        
                        corrs_test = zeros(num_redo,1);
                        rmss_test = zeros(num_redo,1);
                        longCorrTest = zeros(num_redo,1);
                        longRMSTest = zeros(num_redo,1);
                        
                        for f=1:num_redo
                            initial_Theta = randInitializeWeights(input_layer_size, num_alphas);

                            % Training
                            [alphasCCNF, betasCCNF, thetasCCNF] = CCNF_training_bfgs(threshold_x, threshold_fun, train_samples, train_labels, alphas, betas, initial_Theta, lambda_a, lambda_b, lambda_th, similarities, [], 'const', true, 'reinit', reinit, varargin{:});

                            %--- Evaluation on test partition
                            % We don't care about offset and scaling here, as
                            % training data is scaled properly anyway
                            offsets = 0;
                            scalings = 1;
                            [~,~,corrs_test(f), rmss_test(f), longCorrTest(f), longRMSTest(f)] = evaluate_CCNF_model(alphasCCNF, betasCCNF, thetasCCNF, test_samples, test_labels, similarities, [], offsets, scalings, false);

                        end
                        
                        errs_ar = mean(longRMSTest);
                        corrs_ar = mean(longCorrTest);
                        corrs_test_m = mean(corrs_test);
                        rmss_test_m = mean(rmss_test);
                        
                        cv_corr_res(n,a,b,th) = cv_corr_res(n,a,b,th) + corrs_ar;
                        cv_rms_res(n,a,b,th) = cv_rms_res(n,a,b,th) + errs_ar;
                        cv_short_corr(n,a,b,th) = cv_short_corr(n,a,b,th) + corrs_test_m;
                        cv_short_rms(n,a,b,th) = cv_short_rms(n,a,b,th) + rmss_test_m;
                        
                        fprintf('CCNF corr on test: %.3f, rmse: %.3f ', mean(longCorrTest), mean(longRMSTest));     
                        fprintf('Layers %d, alpha %.4f, beta %.4f, theta %.4f \n', num_alphas, lambda_a, lambda_b, lambda_th);
                    end
                end
            end
        end
    end
    
    cv_corr_res = cv_corr_res / num_cv_folds;
    cv_rms_res = cv_rms_res / num_cv_folds;
    
    cv_short_corr = cv_short_corr / num_cv_folds;
    cv_short_rms = cv_short_rms / num_cv_folds;
    
    %% Finding the best values of regularisations and the number of layers
    [val_ar,~] = min(min(min(min(cv_short_rms))));
    [a, b, c, d] = ind2sub(size(cv_short_rms), find(cv_short_rms == val_ar));
    
    best_lambda_a = cv_lambda_a_vals(a(1), b(1), c(1), d(1));
    best_lambda_b = cv_lambda_b_vals(a(1), b(1), c(1), d(1));
    best_lambda_th = cv_lambda_th_vals(a(1), b(1), c(1), d(1));
    best_num_layer =  cv_num_layers(a(1), b(1), c(1), d(1));
        
    
end

