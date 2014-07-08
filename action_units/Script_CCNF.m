clear

addpath('../CCNF/lib');

num_test_folds = 27;
addpath('C:\liblinear\matlab')

correlations = zeros(num_test_folds, 1);
RMS = zeros(num_test_folds, 1);

predictions = cell(num_test_folds, 1);

gt = cell(num_test_folds, 1);

best_lambdas_a = zeros(num_test_folds,1);
best_lambdas_b = zeros(num_test_folds,1);
best_lambdas_th = zeros(num_test_folds,1);
best_num_layers = zeros(num_test_folds,1);

%% Preparing the similarity functions
% number of similarity functions
nGaussians = 0;
nNeighbor = 1;

%% Some parameters
threshold_x = 1e-8;
threshold_fun = 1e-4;
max_iter = 250;

%%
lambdas_a = [1, 2, 3];
lambdas_b = [0, 1, 2];
lambdas_th = [-2, -1, 0, 1];
neural_layers = [5, 10];

% Load the data and AU definitions
shared_defs;

%% load shared definitions
shared_defs;

users = vid_user';

for a=7:numel(aus)
    
    au = aus(a);
    
    %%
    num_cv_folds = 2;
    
    for test_fold = 1:num_test_folds
        
        %% use all but test_fold
        train_users = test_fold;
        test_users = setdiff(1:num_test_folds, test_fold);
        rest_aus = setdiff(aus, au);
        
        [data_train, labels_train, data_valid, labels_valid, data_test, labels_test, Ws, seq_length] = collect_training_data(data_appearance, data_geom, patches_around_landmarks, vid_id, au_patches{a}, au, rest_aus, users, test_users, train_users);

        % normalise to 0 - 1
        labels_test = labels_test/5;
        labels_valid = labels_valid/5;
        labels_train = labels_train/5;
        
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
        
        % this will create a family of exponential decays with different sigmas
        similarities = {};

        range = 1:num_features; % all features used for the exponential similarity

        for i=1:nNeighbor
            neigh = i;
            neighFn = @(x) similarityNeighbor(x, neigh, range);
            similarities = cat(1, similarities, {neighFn});
        end
        
        sparsities = {};

        range = 1:num_features; % all features used for the exponential similarity

        for i=1:0
            neigh = i;
            neighFn = @(x) similarityNeighbor(x, neigh, range);
            sparsities = cat(1, sparsities, {neighFn});
        end        
        reinit = true;
        
        [ best_lambda_a, best_lambda_b, best_lambda_th, best_num_layer,...
            cv_corr_res, cv_rms_res, cv_short_corr, cv_short_rms, ~, ~, ~, ~ ] = ...
            Crossvalidate_CCNF(data_train, labels_train, data_valid, labels_valid, similarities, sparsities, ...
            threshold_x, threshold_fun, max_iter, lambdas_a, lambdas_b, lambdas_th, neural_layers, reinit);
                
        % save the information about the current fold        
        best_lambdas_a(test_fold) = best_lambda_a;
        best_lambdas_b(test_fold) = best_lambda_b;
        best_lambdas_th(test_fold) = best_lambda_th;
        best_num_layers(test_fold) = best_num_layer;
%         
        %%
        initial_Theta = randInitializeWeights(num_features, best_num_layers(test_fold));
        alphas_in = 1 * ones(best_num_layers(test_fold),1);
        betas_in = 1 * ones(numel(similarities)+numel(sparsities), 1);

        data_train = cat(1, data_train, data_valid);
        labels_train = cat(1, labels_train, labels_valid);
        
        [alphas, betas, thetas, likelihoods] = CCNF_training_bfgs(threshold_x, threshold_fun, data_train, labels_train, alphas_in, betas_in, initial_Theta, best_lambda_a, best_lambda_b, best_lambda_th, similarities,sparsities, 'lbfgs', 'const',true, 'reinit', true, 'max_iter', max_iter);

        fprintf('-------------------------------------------------------\n');
        
        [~, ~,~, ~, ~, ~, pred, gts ] = evaluate_CCNF_model(alphas, betas, thetas, data_test, labels_test, similarities, sparsities, 0, 1, false);
        correlations(test_fold) = corr(gts, pred);
        RMS(test_fold) = sqrt(mean((gts*5 - pred*5).^2));
        
        fprintf('CCNF corr on test: %.3f, rmse: %.3f (%d, %d, %f, %d) \n', correlations(test_fold), RMS(test_fold), best_lambda_a, best_lambda_b, best_lambda_th, best_num_layers(test_fold));
                
        predictions{test_fold} = pred;
        gt{test_fold} = gts;
        
        fprintf('--------------------------------- %d done ---------------------\n', test_fold);
        pause(0);
        toc;
    end
    time_taken = toc;
    
    [ accuracies, F1s, corrs, rms, classes ] = evaluate_classification_results( 5*cat(1,predictions{:}), 5*cat(1,gt{:}) );
        
    %%
    name = sprintf('results/Res_CCNF_%d', au);
    save(name, 'best_lambdas_a', 'best_lambdas_b', 'best_lambdas_th', 'best_num_layers', ...
        'accuracies', 'F1s', 'corrs', 'rms', 'correlations', 'time_taken', 'predictions', 'gt');
end