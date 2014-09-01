clear

% Change to your downloaded location
addpath('C:\liblinear\matlab')

num_test_folds = 27;

correlations = zeros(num_test_folds, 1);
RMS = zeros(num_test_folds, 1);

predictions = cell(num_test_folds, 1);

gt = cell(num_test_folds, 1);

best_cs = zeros(num_test_folds,1);
best_ps = zeros(num_test_folds,1);
best_gs = zeros(num_test_folds,1);

%% load shared definitions and AU data
shared_defs;

%%
for a=1:numel(aus)
    
    au = aus(a);
    
    tic
    for test_fold = 1:num_test_folds
        
        %% use all but test_fold
        train_users = test_fold;
        test_users = setdiff(1:num_test_folds, test_fold);
        rest_aus = setdiff(aus, au);
        
        [data_train, labels_train, data_valid, labels_valid, data_test, labels_test, ~, seq_length] = collect_training_data(data_appearance, data_geom, patches_around_landmarks, vid_id, au_patches{a}, au, rest_aus, users, test_users, train_users, out_dir);
        
        data_train = data_train';
        data_valid = data_valid';
        data_test = data_test';
        
        %% Cross-validate here
        [best_c, best_p, corrs, rmss,...
            cs,ps] = Crossvalidate_SVR_lin(data_train, labels_train, data_valid, labels_valid, true);

        %% The actual testing results
        
        % Combine training and validation data
        data_train = cat(1, data_train, data_valid);
        labels_train = cat(1, labels_train, labels_valid);        
        
        % add the bias term
        data_train_b = cat(2, data_train, ones(size(data_train, 1),1));
        data_test_b = cat(2, data_test, ones(size(data_test, 1),1));
        
        comm = sprintf('-s 11 -p %f -c %f -q', best_p, best_c);
        
        model = train(labels_train, sparse(data_train_b), comm);
        
        prediction = predict(labels_test, sparse(data_test_b), model);

        predictions{test_fold} = prediction;

        gt{test_fold} = labels_test;

        correlations(test_fold) = corr(labels_test, prediction);
        RMS(test_fold) = sqrt(mean((labels_test - prediction).^2));
        
        best_cs(test_fold) = best_c;
        best_ps(test_fold) = best_p;

        fprintf('-----------------------------------------------\n');
        fprintf('Results on fold %d: corr - %f, rmse - %f\n', test_fold,  correlations(test_fold), RMS(test_fold));
        fprintf('Best p - %f, best - c - %f\n', best_p, best_c);
        fprintf('-----------------------------------------------\n');
    end
    time_taken = toc;
    short_correlations = mean(correlations);

    name = sprintf('results/Res_SVR_lin_%d', au);

    [ accuracies, F1s, corrs, rms, classes ] = evaluate_classification_results( cat(1,predictions{:}), cat(1,gt{:}) );    
      
    save(name, 'best_cs', 'best_ps', 'accuracies', 'F1s', 'corrs', 'rms', 'correlations',...
               'time_taken', 'predictions', 'gt');
end
