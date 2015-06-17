function [ patch_expert_mirrored ] = Mirror_DCCNF_expert( patch_expert )
%MIRROR_DCCNF_EXPERT Summary of this function goes here
%   Detailed explanation goes here
    patch_expert_mirrored = patch_expert;
    num_hl = size(patch_expert.thetas{1}, 1);
    num_mod = size(patch_expert, 3);
    for m=1:num_mod
        for hl=1:num_hl
            w = reshape(patch_expert.thetas{1}(hl, 2:end, m),11,11);
            w = fliplr(w);
            w = reshape(w, 121,1);
            patch_expert_mirrored.thetas{1}(hl, 2:end, m) = w;
        end
    end
end

