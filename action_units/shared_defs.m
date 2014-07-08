% this is data defined across the experiments (to make sure all of them use
% the same patch experts and have same user conventions)

% Defining which AU's we are extracting and which patches around landmarks
% to extract for them
aus = [1,2,4,5,6,9,12,15,17,20,25,26];

au_patches = {[20,22,23,25,28,40,43];
    [18,20,22,23,25,27,28,40,43];
    [18,20,22,23,25,27,28,40,43];
    [38,40,43,45];
    [37,49,55,46];
    [29,34];
    [49,52,55,58];
    [49,52,55,58];
    [49,52,55,58];
    [49,52,55,58];
    [49,52,55,58];
    [9,49,52,55,58];};

%% 
addpath('./data_preparation/');

% load all of the data together (for efficiency)
% it will be split up accordingly at later stages
if(exist('F:/datasets/DISFA/', 'file'))
    DISFA_dir = 'F:/datasets/DISFA/';
elseif(exist('D:/Databases/DISFA/', 'file'))        
    DISFA_dir = 'D:/Databases/DISFA/';
elseif(exist('Z:/datasets/DISFA/', 'file'))        
    DISFA_dir = 'Z:/Databases/DISFA/';
else
    fprintf('DISFA location not found (or not defined)\n'); 
end

out_dir = [DISFA_dir, '/eccv_features/'];

patch_size = 25;
scale = 0.75;

landmarks_to_extract = sort(unique(cat(2, au_patches{:})));

[data_loc] = extract_data(DISFA_dir, out_dir, patch_size, scale, landmarks_to_extract);

load(data_loc);

users = vid_user';