clear

num_test_folds = 27;

% Change to your liblinear location
addpath('C:\liblinear\matlab')

% CCRF library
addpath('../CCRF/lib');

correlations = zeros(num_test_folds, 1);
long_correlations = zeros(num_test_folds,1);

RMS = zeros(num_test_folds, 1);
long_RMS = zeros(num_test_folds,1);

predictions = cell(num_test_folds, 1);

gt = cell(num_test_folds, 1);

best_cs = zeros(num_test_folds,1);
best_ps = zeros(num_test_folds,1);
best_gs = zeros(num_test_folds,1);

best_lambdas_a = zeros(num_test_folds,1);
best_lambdas_b = zeros(num_test_folds,1);

% Load the data and AU definitions
shared_defs;

%%
for a=1:numel(aus)
    
    au = aus(a);
    
    %%
    tic
    for test_fold = 1:num_test_folds
        
        %% use all but test_fold
        train_users = test_fold;
        test_users = setdiff(1:num_test_folds, test_fold);
        rest_aus = setdiff(aus, au);
        
        [data_train, labels_train, data_valid, labels_valid, data_test, labels_test, Ws, seq_length] = collect_training_data(data_appearance, data_geom, patches_around_landmarks, vid_id, au_patches{a}, au, rest_aus, users, test_users, train_users);

        labels_train = labels_train / 5;
        labels_valid = labels_valid / 5;
        labels_test = labels_test / 5;
        
        data_train = data_train';
        data_valid = data_valid';
        data_test = data_test';             
        
        data_test = cat(2, data_test, ones(size(data_test, 1),1));
        
        % First train the SVR model
                
        %% first train the svr on per frame basis
        
        % cross-validate it
        [best_c, best_p, corrs, rmss,...
            ~,~] = Crossvalidate_SVR_lin(data_train, labels_train, data_valid, labels_valid, true);

        %% Now that the SVR hyper-parameters are found we can train an SVR model
        
        % adding the bias term
        data_train_b = cat(2, data_train, ones(size(data_train, 1),1));
        data_valid_b = cat(2, data_valid, ones(size(data_valid, 1),1));

        comm = sprintf('-s 11 -p %f -c %f -q', best_p, best_c);
        model = train(labels_train, sparse(data_train_b), comm);        
        
        %% use SVR to predict the arousal and valence on the CCRF sample
        svr_prediction_train = predict(labels_train, sparse(data_train_b), model);
        svr_prediction_valid = predict(labels_valid, sparse(data_valid_b), model);
        
        %% Preparing the similarity functions to be used by CCRF
        % number of similarity functions
        nNeighbor = 1;

        % this will create a family of exponential decays with different sigmas
        similarities = {};

        num_features = size(data_train,2);

        range = 1:num_features; % all features used for the exponential similarity

        for i=1:nNeighbor
            neigh = i;
            neighFn = @(x) similarityNeighbor(x, neigh, range);
            similarities = cat(1, similarities, {neighFn});
        end           
        
        samples_ccrf_train = mat2cell(svr_prediction_train, seq_length*ones(1, size(svr_prediction_train,1)/seq_length), 1);        
        samples_ccrf_valid = mat2cell(svr_prediction_valid, seq_length*ones(1, size(svr_prediction_valid,1)/seq_length), 1);        
        
        labels_ccrf_train = mat2cell(labels_train, seq_length*ones(1, size(labels_train,1)/seq_length), 1);
        labels_ccrf_valid = mat2cell(labels_valid, seq_length*ones(1, size(labels_valid,1)/seq_length), 1);
        
        %% Some parameters
        threshold_x = 1e-10;
        threshold_fun = 1e-6;
        max_iter = 200;
        
        lambdas_a = [-3:1];
        lambdas_b = [-1:6];
        % do the arousal bit first
        
        [ best_lambda_a, best_lambda_b, cv_corr_res, cv_rms_res, cv_short_corr, cv_short_rms, ~, ~] = ...
            Crossvalidate_CCRF(samples_ccrf_train, labels_ccrf_train, samples_ccrf_valid, labels_ccrf_valid, similarities, ...
            threshold_x, threshold_fun, lambdas_a, lambdas_b, max_iter);
        
        % save the information about the current fold
        best_lambdas_a(test_fold) = best_lambda_a;
        best_lambdas_b(test_fold) = best_lambda_b;
        
        %% Finally do the testing
        
        alphas = 1 * ones(1,1);
        betas = 1 * ones(numel(similarities), 1);
        
        % train on the whole set now
        
        training_samples = cat(1, svr_prediction_train, svr_prediction_valid);
        
        training_samples = mat2cell(training_samples, seq_length*ones(1, size(training_samples,1)/seq_length), 1);        
        labels_train = cat(1, labels_train, labels_valid);
        
        training_labels = mat2cell(labels_train, seq_length*ones(1, size(labels_train,1)/seq_length), 1);
        
        n_examples = numel(training_samples);
        
        [alphas, betas, scaling, ~] = ...
            CCRF_training_bfgs(n_examples, threshold_x, threshold_fun, training_samples, training_labels, training_labels, alphas, betas, best_lambda_a, best_lambda_b, similarities);

        % Do svr prediction on the test set
        prediction = predict(labels_test, sparse(data_test), model);
        % Convert them to cell represenation
        prediction = {prediction};

        % calculate the offsets for the prediction
        x_offsets = zeros(size(labels_test));
        
        % Evaluate on the test set
        [correlations(test_fold), RMS(test_fold),~, ~,long_correlations(test_fold), long_RMS(test_fold), pred, gts ] = ...
            evaluateCCRFmodel(alphas, betas, prediction, x_offsets, {labels_test}, similarities, scaling, false);
        fprintf('CCRF corr on test: %.3f, rmse: %.3f \n', long_correlations(test_fold), long_RMS(test_fold));
        
        correlations(test_fold) = corr(pred, gts);
        RMS(test_fold) = sqrt(mean((pred*5 - gts*5).^2));
        
        predictions{test_fold} = pred * 5;
        gt{test_fold} = gts * 5;        
        
        fprintf('--------------------------------- %d done ---------------------\n', test_fold);
        
        
    end
    time_taken = toc;
    
    [ accuracies, F1s, corrs, rms, classes ] = evaluate_classification_results( cat(1,predictions{:}), cat(1,gt{:}) );    
    
    %%
    name = sprintf('results/Res_CCRF_%d', au);
    save(name, 'best_lambdas_a', 'best_lambdas_b', ...
        'accuracies', 'F1s', 'corrs', 'rms', 'long_RMS', 'long_correlations', ...
        'time_taken', 'predictions', 'gt');
end