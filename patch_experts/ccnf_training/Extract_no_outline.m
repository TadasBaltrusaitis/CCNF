addpath('../../CCNF/lib');
name = 'ccnf_patches_1_general';
scaling = 1;

no_out_name = [name, '_no_out'];

inner_inds = 18:68;

load(['trained/', name]);

% output the training
locationTxtCol = sprintf('trained/%s.txt', no_out_name);
locationMlabCol = sprintf('trained/%s.mat', no_out_name);

visiIndex = visiIndex(:, inner_inds);

patch_experts.correlations = patch_experts.correlations(:,inner_inds);
patch_experts.rms_errors = patch_experts.rms_errors(:,inner_inds);
patch_experts.patch_experts = patch_experts.patch_experts(:,inner_inds);

Write_patch_experts_ccnf(locationTxtCol, locationMlabCol, scaling, centers, visiIndex, patch_experts, normalisationOptions, [7,9,11,15]);
