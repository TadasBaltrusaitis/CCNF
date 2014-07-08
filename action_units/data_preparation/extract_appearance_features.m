% This function extracts the appearance and geometric features from the
% DISFA dataset (might be slow due to the need to process these features)
function [data_appearance, data_geom, vid_id, vid_user] = extract_appearance_features(DISFA_dir, fp_dir, patch_size, face_scale, landmarks_of_interest)

addpath('./data_preparation/PDM_helpers');

vid_dir = [DISFA_dir, '/Videos_LeftCamera/'];

% First extract the feature point locations accross the datasets (as they
% are in separate files)
[feature_points, vid_names] = extract_landmarks(DISFA_dir, fp_dir);

% Now we can collect the appearance features from videos
vids = dir([vid_dir '*.avi']);

num_patches = numel(landmarks_of_interest);

data_appearance = uint8([]);
data_geom = [];
vid_id = [];

% We use a 66 point Point Distribution Model constructed from Multi-PIE

load('pdm_66_multi_pie.mat');

num_points = 66;

ref_shape = face_scale * [M(1:num_points), M(num_points+1:num_points*2)];

% Some predefinitions for faster patch extraction
widthImg = 1024;
heightImg = 768;
[xi, yi] = meshgrid(1:widthImg,1:heightImg);
xi = double(xi);
yi = double(yi);

for v=1:numel(vids)
        
    input_video = [vid_dir, vids(v).name];
    
    vr = VideoReader(input_video);

    num_frames = vr.NumberOfFrames;
            
    % initialise patch space
    appearance_feats_vid = uint8(zeros(patch_size * patch_size, num_frames, num_patches));
    geom_feats_vid = zeros(numel(E), num_frames);
    vid_id_vid = v * ones(num_frames,1);
    
    % Find the correct landmark locations for this video
    name = vids(v).name(10:14);
    
    vid_user{v} = name;
    
    lmark_inds = strfind(vid_names, name);
    lmark_ind = -1;
    for i = 1:numel(lmark_inds)
        if(~isempty(lmark_inds{i}))
            lmark_ind = i;
            break;
        end
    end
    
    feature_points_current = feature_points{lmark_ind};
    
    % if this version throws a "Dot name reference on non-scalar structure"
    % error change obj.NumberOfFrames to obj(1).NumberOfFrames (in two
    % places in read function) or surround it with an empty try catch
    % statement
        
    step_size = 600;
    
    frame = 0;
    
    % Read frames in 100s for speed
    for i=1:step_size:num_frames
        
        num_to_read = min(num_frames - i + 1, step_size);
        
        all_frames = read(vr, [i, i+num_to_read-1]);
    
        % Extract appearance and geometry features from each frame
        for f=1:num_to_read

            
            frame = frame + 1;
            
            landmark_locations = feature_points_current(i+f-1, :);
            landmark_locations = [landmark_locations(1:num_points)', landmark_locations(num_points+1:end)'];
            curr_frame = all_frames(:,:,:,f);
            curr_frame = double(rgb2gray(curr_frame));
            
            % first get geometry
            [ ~, ~, ~, ~, params] = fit_PDM_ortho_proj_to_2D(M, E, V, landmark_locations);
            geom_feats_vid(:,frame) = params;
            
            % Work out the similarity transform
            [A_img2ref, ~, ~, ~] = AlignShapesWithScale(landmark_locations, ref_shape);

            % transform the current shape to the reference one, so we can
            % interpolate
            shape2D_in_ref = (A_img2ref * (landmark_locations+1)')';

            sideSizeX = (patch_size - 1)/2;
            sideSizeY = (patch_size - 1)/2;

            patches = zeros(patch_size * patch_size, numel(landmarks_of_interest));

            Ainv = inv(A_img2ref);

            % extract patches on which patch experts will be evaluted
            p_curr = 0;
            for l=landmarks_of_interest
                p_curr = p_curr + 1;
                xs = (shape2D_in_ref(l,1)-sideSizeX):(shape2D_in_ref(l,1)+sideSizeX);
                ys = (shape2D_in_ref(l,2)-sideSizeY):(shape2D_in_ref(l,2)+sideSizeY);                

                [xs, ys] = meshgrid(xs, ys);

                pairs = [xs(:), ys(:)];

                actualLocs = (Ainv * pairs')';

                actualLocs(actualLocs(:,1) < 1,1) = 1;
                actualLocs(actualLocs(:,2) < 1,2) = 1;
                actualLocs(actualLocs(:,1) > widthImg,1) = widthImg;
                actualLocs(actualLocs(:,2) > heightImg,2) = heightImg;

                [t_patch] = interp2_mine(xi, yi, curr_frame, actualLocs(:,1), actualLocs(:,2), 'bilinear');
                patches(:,p_curr) = t_patch(:);

            end            
            appearance_feats_vid(:, frame,:) = uint8(patches);
            
            if(mod(frame, 200) == 0)
               fprintf('Vid %d, frame %d done\n',  v, frame);
            end
            
        end
        
        
    end

    data_appearance = cat(2, data_appearance, appearance_feats_vid);
    data_geom = cat(2, data_geom, geom_feats_vid);
    vid_id = cat(1, vid_id, vid_id_vid);
end

end