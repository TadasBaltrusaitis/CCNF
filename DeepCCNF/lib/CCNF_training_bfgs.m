function [ alphas, betas, thetas, final_likelihood] = CCNF_training_bfgs(thresholdX, thresholdFun, x, y, x_v, y_v, alphas, betas, thetas, lambda_a, lambda_b, lambda_th, similarityFNs, sparsityFNs,  networkConfig, varargin)
%CCNF_training_bfgs Performs CCNF training using BFGS (or LBFGS)

    if(sum(strcmp(varargin,'const')))
        ind = find(strcmp(varargin,'const')) + 1;
        const = varargin{ind};        
    else        
        const = false;
    end
    
    cell_x=x;
    cell_y=y;
    
    if(iscell(x))        
        num_seqs = numel(x);
        num_seqs_dev = numel(x);   
        
        x = cell2mat(x)';
        % add a bias term
        x =  cat(1, ones(1,size(x,2)), x);        
        
        %making the dev set
        x_v=cell2mat(x_v)';
        x_v =  cat(1, ones(1,size(x_v,2)), x_v);  
        
        
        % If all of the sequences are of the same length can flatten them
        % to the same matrix
        if(const)
            
            y = cell2mat(y);
            y = reshape(y, numel(y)/num_seqs, num_seqs);
            
            y_v= cell2mat(y_v);
            y_v = reshape(y_v, numel(y_v)/num_seqs_dev, num_seqs_dev);
            
        end
        
    else
        % if not a cell it has already been flattened, and is constant
        % (most likely)
        num_seqs = varargin{find(strcmp(varargin, 'num_seqs'))+1};
        num_seqs_dev = varargin{find(strcmp(varargin, 'num_seqs_dev'))+1};
    end
    
    % Should try a bunch of seed for initialising theta?
    if(sum(strcmp(varargin,'reinit')))
        ind = find(strcmp(varargin,'reinit')) + 1;
        reinit = varargin{ind};
    else
        reinit = false; 
    end
        
    % It is possible to predefine the components B^(k) and C^(k) required 
    % to compute B and and C terms and partial derivative (from equations 
    % 30 and 31 in Appendix B), also can predefine yB^(k)y and yC^(k)y,
    % as they also do not change through the iterations
    % In constant case Precalc_Bs are same across the sequences, same for 
    % PrecalcBsFlat, however yB^(k)y is defined per sequence
    if(sum(strcmp(varargin,'PrecalcBs')) && sum(strcmp(varargin,'PrecalcBsFlat'))...
             && sum(strcmp(varargin,'Precalc_yBy')))
        ind = find(strcmp(varargin,'PrecalcBs')) + 1;
        Precalc_Bs = varargin{ind};

        ind = find(strcmp(varargin,'PrecalcBsFlat')) + 1;
        Precalc_Bs_flat = varargin{ind};

        ind = find(strcmp(varargin,'Precalc_yBys')) + 1;
        Precalc_yBys = varargin{ind};
    else
        % if these are not provided calculate them        
        [ ~, Precalc_Bs, Precalc_Bs_flat, Precalc_yBys ] = CalculateSimilarities( num_seqs, x, similarityFNs, sparsityFNs, y, const);
    end
    %calc similarities for dev set anyways!
    
    [ ~, Precalc_Bs_v, Precalc_Bs_flat_v, Precalc_yBys_v ] = CalculateSimilarities( num_seqs_dev, x_v, similarityFNs, sparsityFNs, y_v, const);
    
    
    % Reinitialisation attempts to find a better starting point for the
    % model training (sometimes helps sometimes doesn't)
    if(reinit)
        
        rng(0);
        
        % By default try 300 times, but can override
        num_reinit = 300;
        
        if(sum(strcmp(varargin,'num_reinit')))
            num_reinit = varargin{find(strcmp(varargin,'num_reinit')) + 1};
        end
        
        thetas_good = cell(num_reinit, 1);
        lhoods = zeros(num_reinit, 1);
        for i=1:num_reinit
            initial_Theta = randInitializeWeights(networkConfig);            
            lhoods(i) = LogLikelihoodCCNF(y, x, alphas, betas, initial_Theta, lambda_a, lambda_b, lambda_th, Precalc_Bs_flat, [], [], [], [], const, num_seqs);
            thetas_good{i} = initial_Theta;
        end
        [~,ind_max] = max(lhoods);
        thetas = thetas_good{ind_max};
    end
    
    thetas=thetaInVector(thetas);
    
    params = [alphas; betas; thetas(:)];
    
    %fuck this part! 
    bigNumber=1000;
    eta=ones(size(params))/500;
    eta(1:size(alphas,1)+size(betas,1))=eta(1:size(alphas,1)+size(betas,1))/bigNumber;
    numEpochs=500; % smallish for now
    percentageForBatch=0.1;
    percentageForDropout=.5;
    counterForCalcLiklihood=100;
    liklihoodTicks=1/percentageForBatch;
    %till here!
    for j=1:numEpochs
        if(iscell(cell_x))            
            num_seqs = numel(cell_x);
        end
        
        num_seqs_in_batch=ceil(num_seqs*percentageForBatch);
        randSequences=randsample(num_seqs,num_seqs_in_batch);
                  
        mask=makeMask(networkConfig,percentageForDropout);
        
        if(iscell(cell_x))
            xr=cell_x(randSequences);
            xr = cell2mat(xr)';
            xr =  cat(1, ones(1,size(xr,2)), xr);
        else
            % Pick the right sequences from ids
            seq_length = size(x, 2) / num_seqs;
            seq_ids = zeros(num_seqs, 1);
            seq_ids(randSequences) = 1;
            seq_ids = (seq_ids * ones(1, seq_length))';
            seq_ids = logical(seq_ids(:));
            xr = cell_x(:, seq_ids);
        end
        
        if(const)
            if(iscell(cell_y))                
                yr = cell2mat(cell_y(randSequences));
                yr = reshape(yr, numel(yr)/num_seqs_in_batch, num_seqs_in_batch);
            else
                yr = cell_y(:, sort(randSequences));
            end

        end
        
        [ ~, Precalc_Bs_r, Precalc_Bs_flat_r, Precalc_yBys_r ] = CalculateSimilarities( num_seqs_in_batch, xr, similarityFNs, sparsityFNs, yr, const);
        [params,likelihood]=gradientDecentOneEpoch(params, numel(alphas), numel(betas), size(thetas), lambda_a, lambda_b, lambda_th, Precalc_Bs_r, xr, yr, Precalc_yBys_r, Precalc_Bs_flat_r, const, networkConfig,eta,mask);
        
        %eta decay
        eta=eta*0.9999;
        counterForCalcLiklihood=counterForCalcLiklihood+1;
        if(mod(counterForCalcLiklihood,liklihoodTicks)==0)
        	alphas = params(1:numel(alphas));
            betas = params(numel(alphas)+1:numel(alphas)+numel(betas));
            thetas_temp = thetaInCell(params(numel(alphas) + numel(betas) + 1:end), networkConfig);
    
            for i=1:size(thetas_temp,2)-1
                thetas_temp{i}=thetas_temp{i}*percentageForDropout;
            end
            onTrain = LogLikelihoodCCNF(y, x, alphas, betas, thetas_temp, lambda_a, lambda_b, lambda_th, Precalc_Bs_flat, [], [], [], [], const, num_seqs);
            onDev=LogLikelihoodCCNF(y_v, x_v, alphas, betas, thetas_temp, lambda_a, lambda_b, lambda_th, Precalc_Bs_flat_v, [], [], [], [], const, num_seqs);
            fprintf('%d: - On train data:%.3f, on dev data %.3f\n', j, onTrain, onDev);
        end
    end
    
    alphas = params(1:numel(alphas));
    betas = params(numel(alphas)+1:numel(alphas)+numel(betas));
    thetas = thetaInCell(params(numel(alphas) + numel(betas) + 1:end), networkConfig);
    
    for i=1:size(thetas,2)-1
        thetas{i}=thetas{i}*percentageForDropout;
    end
    

    final_likelihood = LogLikelihoodCCNF(y, x, alphas, betas, thetas, lambda_a, lambda_b, lambda_th, Precalc_Bs_flat, [], [], [], [], const, num_seqs);

end

function [loss, gradient] = objectiveFunction(params, numAlpha, numBeta, sizeTheta, lambda_a, lambda_b, lambda_th, PrecalcQ2s, x, y, PrecalcYqDs, PrecalcQ2sFlat, const,networkConfig)
    
    alphas = params(1:numAlpha);
    betas = params(numAlpha+1:numAlpha+numBeta);
    thetas = thetaInCell(params(numAlpha + numBeta + 1:end), networkConfig);
    
    
    
    num_seqs = size(PrecalcYqDs,1);
    
    [gradient, SigmaInvs, CholDecomps, Sigmas, bs, all_x_resp] = gradientCCNF(params, numAlpha, numBeta, sizeTheta, lambda_a, lambda_b, lambda_th, PrecalcQ2s, x, y, PrecalcYqDs, PrecalcQ2sFlat, const, num_seqs,networkConfig);
    
    %checkWeights(gradient,x,y,alphas,betas,thetas,lambda_a, lambda_b, lambda_th,PrecalcQ2sFlat,const, num_seqs,networkConfig);
    
    % as bfgs does gradient descent rather than ascent, negate the results
    gradient = -gradient;
    loss = -LogLikelihoodCCNF(y, x, alphas, betas, thetas, lambda_a, lambda_b, lambda_th, PrecalcQ2sFlat, SigmaInvs, CholDecomps, Sigmas, bs, const, num_seqs, all_x_resp);
end
