function [ predict_labels ] = Predict_GP( train_samples, train_labels, test_samples, test_labels, add_bias )
%CROSSVALIDATE_SVR Summary of this function goes here
%   Detailed explanation goes here

    % adding a bias term? TODO is this needed?
    if(add_bias)
        train_samples = cat(2, train_samples, ones(size(train_samples, 1),1));
    end
        
    % adding the path
    addpath(genpath('../GP'));
        
    %% SAY WHICH CODE WE WISH TO EXERCISE
    id = [1,1]; % use Gauss/Exact
    % id = [1,2; 3,2; 4,2]; % compare Laplace
    % id = [1,3; 2,3; 3,3]; % study EP
    % id = [1,5; 2,5]; % look into KL (takes quite a while)
    % id = [1,4; 2,4; 3,4; 4,4]; % deal with VB

    % TODO maybe can validate the GP types as well
    % TODO should sf, ell, a and b be validated?
          
    cov = {@covSEiso}; sf = 1; ell = 0.4;                             % setup the GP
    hyp0.cov  = log([ell;sf]);
    mean = {@meanSum,{@meanLinear,@meanConst}}; as = 1/5 * ones(size(train_samples,2),1); b = 1;       % m(x) = a*x+b
    hyp0.mean = [as;b];

    sn = 0.2;
    
    lik_list = {'likGauss','likLaplace','likSech2','likT'};   % possible likelihoods
    inf_list = {'infExact','infLaplace','infEP','infVB','infKL'};   % inference algs

    Ncg = 50;                                   % number of conjugate gradient steps
    
    nlZ(1) = -Inf;
    
    for i=1:size(id,1)
        
        lik = lik_list{id(i,1)};
        
        % setup the likelihood
        if strcmp(lik,'likT')
            nu = 4;
            hyp0.lik  = log([nu-1;sqrt((nu-2)/nu)*sn]);
        else
            hyp0.lik  = log(sn);
        end
        
        inf = inf_list{id(i,2)};
        fprintf('OPT: %s/%s\n',lik_list{id(i,1)},inf_list{id(i,2)})
        if Ncg==0
            hyp = hyp0;
        else
            hyp = minimize(hyp0,'gp', -Ncg, inf, mean, cov, lik, train_samples, train_labels); % opt hypers
        end
        
        % predict
        [predict_labels, predict_uncert] = gp(hyp, inf, mean, cov, lik, train_samples, train_labels, test_samples);
        
        %[nlZ(i+1)] = gp(hyp, inf, mean, cov, lik, xtr, ytr);
    end

end

