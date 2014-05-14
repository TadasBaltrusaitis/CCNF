function [ gradientParams, SigmaInvs, CholDecomps, Sigmas ] = gradientCCRFFull( params, lambda_a, lambda_b, PrecalcQ2s, x, y, masks, PrecalcYqDs, useIndicators, PrecalcQ2sFlat)
%GRADIENTPRF Summary of this function goes here
%   Detailed explanation goes here

    nExamples = numel(x);

    numBetas = size(PrecalcQ2sFlat{1},2);
    numAlphas = numel(params) - numBetas;
    
    alphasInit = params(1:numAlphas);
    betasInit = params(numAlphas+1:end);
    gradientParams = zeros(size(params));
    
    % These might be use to calculate the LogLikelihood, don't want to
    % recompute them
    SigmaInvs = cell(nExamples, 1);
    CholDecomps = cell(nExamples, 1);
    Sigmas = cell(nExamples, 1);
    gradients = zeros(nExamples, numel(params));
    for q = 1 : nExamples

        yq = y{q};
        xq = x{q};
        mask = masks{q};

        PrecalcQ2 = PrecalcQ2s{q};
        PrecalcQ2Flat = PrecalcQ2sFlat{q};
        
        [ logGradientsAlphas, logGradientsBetas, SigmaInv, CholDecomp, Sigma ] = gradientCCRF_withoutReg(alphasInit, betasInit, PrecalcQ2, xq, yq, mask, PrecalcYqDs(q, :), useIndicators, PrecalcQ2Flat);
        SigmaInvs{q} = SigmaInv;
        CholDecomps{q} = CholDecomp;
        Sigmas{q} = Sigma;
        
        gradients(q,:) = [logGradientsAlphas; logGradientsBetas];
    end
    gradientParams = sum(gradients,1)';
    regAlpha = alphasInit * lambda_a;
    regBeta = betasInit * lambda_b;
    gradientParams = gradientParams - [regAlpha; regBeta];
end