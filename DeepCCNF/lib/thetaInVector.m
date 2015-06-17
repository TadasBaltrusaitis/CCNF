function [ outputTheta ] = thetaInVector( inputTheta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    outputTheta=[];
    for i=1:size(inputTheta,2)
        outputTheta=cat(1,outputTheta,inputTheta{i}(:));
    end
end

