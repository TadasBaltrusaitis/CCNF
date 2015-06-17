function [ postAc, lastLayerResponse] = feedForward( thetasInit ,x, varargin )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    dropout=0;
    if(size(varargin,2)>0)
        mask=varargin{1};
        dropout=1;
    end
    postAc=cell(size(thetasInit));
    %the first layer is the X. Don't argue with me! I consider it as a
    %post-activation cause it's easier to code that way :P
    
    numLayers=size(thetasInit,2);
    
    
    
    %would be nice to define an activation func for more general cases.
    %Sigmoid only used. 
    for i=1:numLayers;
        if(i~=1)
            biasVectorSize=size(postAc{i-1},2);
            postAc{i}=1./(1+exp(-thetasInit{i}*cat(1,postAc{i-1},repmat([1],1,biasVectorSize))));
        else
            postAc{i}=1./(1+exp(-thetasInit{i}*x));
        end
        %dropout except for last layer!
        if(dropout==1 && i~=numLayers)
            postAc{i}=postAc{i}.*repmat(mask{i}',1,size(x,2));
        end
    end
    lastLayerResponse=postAc{size(postAc,2)};
end

