function [ mask ] = makeMask(networkConfig,percentage)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    mask=cell(size(networkConfig));
    for i=1:size(mask,2)
        sizeThisAcs=networkConfig{i}(1);
        if(i~=size(mask,2))
            mask{i}=(randperm(sizeThisAcs)/sizeThisAcs)<=percentage;
        else
            mask{i}=ones(1,sizeThisAcs);
    end
end

