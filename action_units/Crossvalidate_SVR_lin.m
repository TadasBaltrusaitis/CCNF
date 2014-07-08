function [ best_c, best_p, cv_corr_res, cv_rms_res, cv_c_vals, cv_p_vals] = Crossvalidate_SVR_lin( samples, labels, samples_valid, labels_valid, add_bias )
%CROSSVALIDATE_SVR Summary of this function goes here
%   Detailed explanation goes here

    cs = [-6:2:3];
    ps = [-6:2:3];

    % keep track of the cross-validation hyper-parameters
    cv_c_vals = zeros(numel(cs), numel(ps));
    cv_p_vals = zeros(numel(cs), numel(ps));

    % keep track of results for each fold
    cv_corr_res = zeros(numel(cs), numel(ps));
    cv_rms_res = zeros(numel(cs), numel(ps));
    
    % if we have a cell array convert it to a double one
    if(iscell(samples))
        train_samples = cell2mat(samples);
        train_labels = cell2mat(labels);

        test_samples = cell2mat(samples_valid);
        test_labels = cell2mat(labels_valid);            
    else
        train_samples = samples;
        train_labels = labels;

        test_samples = samples_valid;
        test_labels = labels_valid;  
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

            cv_c_vals(c,p) = cCurr;
            cv_p_vals(c,p) = pCurr;

            comm = sprintf('-s 11 -p %f -c %f -q', pCurr, cCurr);
            model = train(train_labels, sparse(double(train_samples)), comm);

%                 prediction_t = train_samples * model.w';
%                 
%                 plot(prediction_t)
%                 hold on;
%                 plot(train_labels, 'r')
%                 hold off;

            % calc error myself
%                 predict(test_labels, sparse(test_samples), model);                
            prediction = test_samples * model.w';

            errs = sqrt(mean((prediction - test_labels).^2));
            corrs = corr(test_labels, prediction)^2;

            cv_corr_res(c,p) = cv_corr_res(c,p) + corrs;
            cv_rms_res(c,p) = cv_rms_res(c,p) + errs;
            fprintf('SVR corr on validation set: %.3f, rmse: %.3f\n', corrs, errs);     
%                 plot(prediction)
%                 hold on;
%                 plot(test_labels, 'r')
%                 hold off;
        end
    end

    %% Finding the best values of c and p

    [val,~] = min(min(cv_rms_res));
    [a, b] = ind2sub(size(cv_rms_res), find(cv_rms_res == val));

    if(~isempty(a) && ~isempty(b) )        
        best_c = cv_c_vals(a(1), b(1));
        best_p = cv_p_vals(a(1), b(1));
    else
        best_c = cv_c_vals(1,1);
        best_p = cv_p_vals(1,1);
    end
end

