function [ best_lambda_a, best_lambda_b, best_lambda_th, best_num_layer,...
           cv_corr_res, cv_rms_res, cv_short_corr, cv_short_rms, ...
           cv_lambda_a_vals, cv_lambda_b_vals, cv_lambda_th_vals, cv_num_layers ] = ...
            Crossvalidate_CCNF(samples_train, labels_train, samples_valid, labels_valid, similarities, sparsities,...
                threshold_x, threshold_fun, max_iter, lambdas_a, lambdas_b, lambdas_th, neural_layers, reinit)
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

    input_layer_size = size(samples_train{1}, 2);

    for n=1:numel(neural_layers)
        for a = 1:numel(lambdas_a)

            num_alphas = neural_layers(n);

            alphas = 1 * ones(num_alphas,1);
            betas = 1 * ones(numel(similarities)+numel(sparsities), 1);

            lambda_a = 10^lambdas_a(a);

            for b = 1:numel(lambdas_b)

                lambda_b = 10^lambdas_b(b);

                for th = 1:numel(lambdas_th)

                    lambda_th  = 10^lambdas_th(th);

                    cv_lambda_a_vals(n, a, b, th) = lambda_a;
                    cv_lambda_b_vals(n, a, b, th) = lambda_b;
                    cv_lambda_th_vals(n, a, b, th) = lambda_th;
                    cv_num_layers(n, a, b, th) = num_alphas;

                    initial_Theta = randInitializeWeights(input_layer_size, num_alphas);

                    % Training
                    tic
                    [alphasCCNF, betasCCNF, thetasCCNF] = CCNF_training_bfgs(threshold_x, threshold_fun, samples_train, labels_train, alphas, betas, initial_Theta, lambda_a, lambda_b, lambda_th, similarities, sparsities, 'lbfgs', 'const', true, 'reinit', reinit, 'max_iter', max_iter, 'num_reinit', 20);
                    toc
                    %--- Evaluation on test partition
                    offsets = 0;
                    scalings = 1;

                    [~,~,~, ~, corrs, rms, ~, ~] = evaluate_CCNF_model(alphasCCNF, betasCCNF, thetasCCNF, samples_valid, labels_valid, similarities, sparsities, offsets, scalings, false);

                    cv_corr_res(n,a,b,th) = cv_corr_res(n,a,b,th) + corrs;
                    cv_rms_res(n,a,b,th) = cv_rms_res(n,a,b,th) + rms;
                    
                    fprintf('CCNF corr on validation: %.3f, rmse: %.3f ', corrs, rms);     
                    fprintf('Layers %d, alpha %.4f, beta %.4f, theta %.4f \n', num_alphas, lambda_a, lambda_b, lambda_th);
                end
            end
        end
    end
    
    %% Finding the best values of regularisations and the number of layers
    [val,~] = min(min(min(min(cv_rms_res))));
    [a, b, c, d] = ind2sub(size(cv_rms_res), find(cv_rms_res == val));
    
    best_lambda_a = cv_lambda_a_vals(a(1), b(1), c(1), d(1));
    best_lambda_b = cv_lambda_b_vals(a(1), b(1), c(1), d(1));
    best_lambda_th = cv_lambda_th_vals(a(1), b(1), c(1), d(1));
    best_num_layer =  cv_num_layers(a(1), b(1), c(1), d(1));
        
    
end

