function [ alphas, betas, scaling, finalLikelihood] = CCRF_training_bfgs( nExamples, thresholdX, thresholdFun, x, y, yUnnormed, masks, alphas, betas, lambda_a, lambda_b, similarityFNs, useIndicators, PrecalcQ2s, PrecalcQ2sFlat, PrecalcYqDs)
%GRADIENTDESCENTCCRF Performs CCRF gradient descen given the initial state
%and gradient descent parameters
%   Detailed explanation goes here

    % if these are not provided calculate them
    if(nargin < 14)
        [ ~, PrecalcQ2s, PrecalcQ2sFlat, PrecalcYqDs ] = CalculateSimilarities( nExamples, x, similarityFNs, y );
    end
        
    params = [alphas; betas];
    
    objectiveFun = @(params)objectiveFunction(params, numel(alphas), lambda_a, lambda_b, PrecalcQ2s, x, y, masks, PrecalcYqDs, useIndicators, PrecalcQ2sFlat);
%         A = -eye(numel(params));
%         b = zeros(numel(params),1);
%         options = optimset('Algorithm','interior-point','GradObj','on', 'Display','iter-detailed', 'TolX', 1e-3, 'TolFun', 1e-3, 'Hessian', 'lbfgs', 'Diagnostics', 'on', 'PlotFcns', @optimplotstepsize);
%         options = optimset('Algorithm','interior-point','GradObj','on', 'TolX', threshold, 'TolFun', threshold, 'Hessian', 'bfgs','Diagnostics', 'on', 'Display', 'iter-detailed');
%         options = optimset('DerivativeCheck', 'on', 'Algorithm','interior-point','GradObj','on', 'Hessian', 'bfgs','Display', 'iter-detailed');
%     options = optimset('Algorithm','interior-point','GradObj','on', 'TolX', 1e-2, 'TolFun', 1e-2, 'Hessian', 'bfgs','Display', 'iter-detailed');
    options = optimset('Algorithm','interior-point','GradObj','on', 'TolX', thresholdX, 'TolFun', thresholdFun, 'Hessian', 'bfgs', 'display','off', 'useParallel', 'Always');
%     options = optimset('Algorithm','interior-point','GradObj','on', 'Hessian', 'bfgs', 'display','off');
    params = fmincon(objectiveFun, params, [], [],[],[], zeros(numel(params),1), Inf(numel(params), 1), [], options);
    alphas = params(1:numel(alphas));
    betas = params(numel(alphas)+1:end);

    finalLikelihood = LogLikelihoodCCRF(y, x, masks, alphas, betas, lambda_a, lambda_b, PrecalcQ2sFlat,useIndicators);
%     fprintf('Final log likelihood at iteration; logL %f, learning rate\n', finalLikelihood);
    
    % establish the scaling
    scaling = getScaling2(alphas, betas, x, yUnnormed, masks, PrecalcQ2s, useIndicators);

end

function [loss, gradient] = objectiveFunction(params, numAlpha, lambda_a, lambda_b, PrecalcQ2s, x, y, masks, PrecalcYqDs, useIndicators, PrecalcQ2sFlat)
    
    alphas = params(1:numAlpha);
    betas = params(numAlpha+1:end);
    [gradient, SigmaInvs, CholDecomps, Sigmas] = gradientCCRFFull(params, lambda_a, lambda_b, PrecalcQ2s, x, y, masks, PrecalcYqDs, useIndicators, PrecalcQ2sFlat);
    % as bfgs does gradient descent rather than ascent, negate the results
    gradient = -gradient;
    loss = -LogLikelihoodCCRF(y, x, masks, alphas, betas, lambda_a, lambda_b, PrecalcQ2sFlat, useIndicators, SigmaInvs, CholDecomps, Sigmas);
end
