function [ params, loss ] = gradientDecentOneEpoch(params, numAlpha, numBeta, sizeTheta, lambda_a, lambda_b, lambda_th, PrecalcQ2s, x, y, PrecalcYqDs, PrecalcQ2sFlat, const,networkConfig, eta,mask)
    
    alphas = params(1:numAlpha);
    betas = params(numAlpha+1:numAlpha+numBeta);
    vectorTheta=params(numAlpha + numBeta + 1:end);
    thetas = thetaInCell(params(numAlpha + numBeta + 1:end), networkConfig);
    
    
    num_seqs = size(PrecalcYqDs,1);
    [gradient, SigmaInvs, CholDecomps, Sigmas, bs, all_x_resp] = gradientCCNF(params, numAlpha, numBeta, sizeTheta, lambda_a, lambda_b, lambda_th, PrecalcQ2s, x, y, PrecalcYqDs, PrecalcQ2sFlat, const, num_seqs,networkConfig,mask);
    
    %checkWeights(gradient,x,y,alphas,betas,thetas,lambda_a, lambda_b, lambda_th,PrecalcQ2sFlat,const, num_seqs,networkConfig);
    
    % as bfgs does gradient descent rather than ascent, negate the results
    %gradient = -gradient;
    
    params=params+eta.*gradient;
    
    indiciesLessThanZero=params<0;
    params(indiciesLessThanZero(1:numAlpha+numBeta))=0.0001;
    
    loss = LogLikelihoodCCNF(y, x, alphas, betas, thetas, lambda_a, lambda_b, lambda_th, PrecalcQ2sFlat, SigmaInvs, CholDecomps, Sigmas, bs, const, num_seqs, all_x_resp);
    
end

