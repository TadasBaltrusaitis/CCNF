function [ outputTheta ] = thetaInCell( inputTheta, sizeTheta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    outputTheta={};
    startForTheta=1;
    for i=1:size(sizeTheta,2)
        thisLayer=sizeTheta{i}(1)*(sizeTheta{i}(2)+1);
        outputTheta{i} = reshape(inputTheta(startForTheta:startForTheta+thisLayer-1), sizeTheta{i}(1),sizeTheta{i}(2)+1);        
        startForTheta=startForTheta+thisLayer;
    end
end

