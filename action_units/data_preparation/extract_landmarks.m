% Extracting feature points from DISFA format
function [feature_point_locations, vid_names] = extract_landmarks(DISFA_dir, output_dir)

% A small correction
load([DISFA_dir, '/Landmark_Points/SN031/tmp_frame_lm/l03630_lm.mat']);
save([DISFA_dir, '/Landmark_Points/SN031/tmp_frame_lm/l03631_lm.mat'], 'pts');

root_dir = [DISFA_dir, '/Landmark_Points/'];
dirs = dir([root_dir, 'SN*']);

if(~exist(output_dir, 'file'))
    mkdir(output_dir);
end

feature_point_locations = cell(numel(dirs), 1);
vid_names = cell(numel(dirs), 1);
% Go through each subject video
for d=1:numel(dirs)

    out_file = [output_dir, '/', dirs(d).name, '.mat'];
    vid_names{d} = dirs(d).name;
    
    % if already created ignore
    if(exist(out_file, 'file'))
        load(out_file);
        feature_point_locations{d} = all_pts;
        continue;
    end
    
    dir_c = dir([root_dir, dirs(d).name]);
    dir_c = dir_c(3).name;
    points = dir([root_dir, dirs(d).name '/' dir_c '/*.mat']);

    all_pts = zeros(numel(points),66*2);
    
    % Go through each frame and collect them
    for i=1:numel(points)

        load([root_dir, dirs(d).name '/' dir_c '/' points(i).name]);
        
        all_pts(i,:) = pts(:);

    end
    
    feature_point_locations{d} = all_pts;

    save(out_file, 'all_pts');    
end

end