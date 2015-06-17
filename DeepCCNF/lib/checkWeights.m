function [ output_args ] = checkWeights( gradient, x, y, alphas, betas, initial_Theta, lambda_a, lambda_b, lambda_th,someValueBflat,const, num_seqs,networkConfig)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    

    verySmall=.000000001;
    alphas_init = gradient(1:10);
    betasInit = gradient(10+1:11);
    vectorTheta=gradient(11+1:end);
    
    vectorTheta=thetaInCell(vectorTheta,networkConfig);
    
    
    numerical=cell(size(initial_Theta));
    
    for i=1:size(initial_Theta,2)
        thisSize=size(initial_Theta{i});
        temp=zeros(thisSize);
        for j=1:thisSize(1)
            for k=1:thisSize(2)
                tempTheta=initial_Theta;
                tempTheta{i}(j,k)=tempTheta{i}(j,k)+verySmall;
                one=LogLikelihoodCCNF(y, x, alphas, betas, initial_Theta, lambda_a, lambda_b, lambda_th, someValueBflat, [], [], [], [], const, num_seqs);
                two=LogLikelihoodCCNF(y, x, alphas, betas, tempTheta, lambda_a, lambda_b, lambda_th, someValueBflat, [], [], [], [], const, num_seqs);
                temp(j,k)=abs(((two-one)*(1/verySmall))-vectorTheta{i}(j,k));
            end
        end
        numerical{i}=temp;
    end
     
end

