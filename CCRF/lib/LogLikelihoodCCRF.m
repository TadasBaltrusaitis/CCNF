function logL = LogLikelihoodCCRF(y_coll, x_coll, masks, alphas, betas,...
                                  lambda_a,lambda_b, PrecalcQ2Flat,...
                                  useIndicator, SigmaInvs, ChDecomps, Sigmas)
% Calculating the log likelihood of the CCRF with multi alpha and beta    

Q = numel(y_coll);
logL = 0;
for q=1:Q
    yq = y_coll{q};
    xq = x_coll{q};
    mask = masks{q};
    
    % constructing the sigma inverse
%     [SigmaInv] = CalcSigmaCCRF(alphas, betas, PrecalcQ2{q}, mask);

    n = size(xq, 1);
    % if these are not provided with the call (they might be, as
    % calculation of gradient involves these terms)
    
    b = CalcbCCRF(alphas, xq, mask, useIndicator);
        
    if(nargin < 11)
        [SigmaInv] = CalcSigmaCCRFflat(alphas, betas, n, PrecalcQ2Flat{q}, mask, useIndicator);
        L = chol(SigmaInv);        
        mu = SigmaInv \ b;
    else
        SigmaInv = SigmaInvs{q};
        L = ChDecomps{q};
        Sigma = Sigmas{q};        
        mu = Sigma * b;
    end    

    % normalisation = 1/((2*pi)^(n/2)*sqrt(det(Sigma)));
    % Removing the division by pi, as it is constant
    % normalisation = 1/(sqrt(det(sigma)));
    % flipping around determinant of SigmaInv, as det(inv(Sigma)) = inv(det(Sigma)  
%     normalisation = log(sqrt(det(SigmaInv)));

    % normalisation 2 using Cholesky decomposition
    normalisation2 = sum(log(diag(L))); % no times 2 here as we calculate the square root of determinant

    % probq = normalisation * exp(-0.5 * (y - mu)'*SigmaInv*(y-mu));
    % applying a logarithm to this leads to
%     logLq = log(normalisation) + (-0.5 * (yq - mu)'*SigmaInv*(yq-mu));
    logLq = normalisation2 + (-0.5 * (yq - mu)'*SigmaInv*(yq-mu));
  
    logL = logL + logLq;
    
end

% add regularisation term
logL = logL -lambda_b * (betas'*betas)/2 - lambda_a * (alphas'*alphas)/2;
