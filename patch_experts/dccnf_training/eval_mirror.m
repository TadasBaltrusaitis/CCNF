function [ responses_ccnf_ncc ] = eval_mirror( unnormed_samples, patch_expert, normalisation_options, normalisationRegion, region_length )
%EVAL_MIRROR Summary of this function goes here
%   Detailed explanation goes here

    patch_expert = Mirror_DCCNF_expert(patch_expert);
    % first flip all the unnormed samples
    for i=1:size(unnormed_samples, 1)
       img = reshape(unnormed_samples(i,:), normalisationRegion(1), normalisationRegion(2));
       img = fliplr(img);
       unnormed_samples(i,:) = img(:);
    end

    [responses_ccnf_ncc] = DCCNF_ncc_response(unnormed_samples, patch_expert, normalisation_options, normalisation_options.normalisationRegion, region_length);

    % flip the responses now
    for i=1:size(unnormed_samples, 1)
       
        resp = reshape(responses_ccnf_ncc((i-1)*region_length + 1:i * region_length), sqrt(region_length), sqrt(region_length)); 
        resp = fliplr(resp);
        responses_ccnf_ncc((i-1)*region_length + 1:i * region_length) = resp(:);
    end
    
end

