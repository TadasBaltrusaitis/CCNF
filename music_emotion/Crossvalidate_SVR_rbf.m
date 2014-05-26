function [ best_c, best_p, best_g, cv_corr_res, cv_rms_res, cv_c_vals, cv_p_vals, cv_g_vals ] = Crossvalidate_SVR_rbf( samples, labels, num_cv_folds, add_bias )
%CROSSVALIDATE_SVR Summary of this function goes here
%   Detailed explanation goes here

    %% first get the indices to use for the cross validation
    [ inds_cv_train, inds_cv_test ] = GetFolds(size(labels,1), num_cv_folds);

    cs = -6:3;
    ps = -8:-1;
    gs = -6:1;

    % keep track of the cross-validation hyper-parameters
    cv_c_vals = zeros(numel(cs), numel(ps), numel(gs));
    cv_p_vals = zeros(numel(cs), numel(ps), numel(gs));
    cv_g_vals = zeros(numel(cs), numel(ps), numel(gs));

    % keep track of results for each fold
    cv_corr_res = zeros(numel(cs), numel(ps), numel(gs));
    cv_rms_res = zeros(numel(cs), numel(ps), numel(gs));

    for i = 1:num_cv_folds

        % if we have a cell array convert it to a double one
        if(iscell(samples))
            train_samples = cell2mat(samples(inds_cv_train(:,i),:));
            train_labels = cell2mat(labels(inds_cv_train(:,i),:));
            
            test_samples = cell2mat(samples(inds_cv_test(:,i),:));
            test_labels = cell2mat(labels(inds_cv_test(:,i),:));            
        else
            train_samples = samples(inds_cv_train(:,i),:);
            train_labels = labels(inds_cv_train(:,i),:);
            
            test_samples = samples(inds_cv_test(:,i),:);
            test_labels = labels(inds_cv_test(:,i),:);  
        end

        % adding a bias term
        if(add_bias)
            train_samples = cat(2, train_samples, ones(size(train_samples, 1),1));
            test_samples = cat(2, test_samples, ones(size(test_samples, 1),1));
        end
        
        % Crossvalidate the c, p, and gamma values
        for c = 1:numel(cs)

            cCurr = 10^cs(c);

            for p = 1:numel(ps)

                pCurr = 2^ps(p);

                for g = 1:numel(gs)

                    gCurr  = 2^gs(g);

                    cv_c_vals(c,p,g) = cCurr;
                    cv_p_vals(c,p,g) = pCurr;
                    cv_g_vals(c,p,g) = gCurr;
            
                    comm = sprintf('-s 3 -t 2 -p %f -c %f -g %f -q', pCurr, cCurr, gCurr);
                    model = svmtrain(train_labels, train_samples, comm);

                    % calc error myself
                    prediction = svmpredict(test_labels, test_samples, model);
                    
                    % Using the long correlation
                    corrs = corr(test_labels, prediction)^2;
                    
                    % using the average of per song RMS errors (supposed to
                    % be bes representation)
                    prediction = reshape(prediction, 15, numel(prediction)/15);
                    test_labels_r = reshape(test_labels, 15, numel(test_labels)/15);
                    rms_per_song = sqrt(mean((prediction - test_labels_r).^2));
                    errs = mean(rms_per_song);                    

                    cv_corr_res(c,p,g) = cv_corr_res(c,p,g) + corrs;
                    cv_rms_res(c,p,g) = cv_rms_res(c,p,g) + errs;
                end
            end
        end
    end

    cv_corr_res = cv_corr_res / num_cv_folds;
    cv_rms_res = cv_rms_res / num_cv_folds;

    %% Finding the best values of c and p and g
    [val_ar,~] = min(min(min(cv_rms_res)));
    [a, b, c] = ind2sub(size(cv_rms_res), find(cv_rms_res == val_ar));

    best_c = cv_c_vals(a(1), b(1), c(1));
    best_p = cv_p_vals(a(1), b(1), c(1));
    best_g = cv_g_vals(a(1), b(1), c(1));

end

