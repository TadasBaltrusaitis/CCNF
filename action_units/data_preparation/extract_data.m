function [data_loc] = extract_data(DISFA_dir, output_dir, patch_size, scale, landmarks_of_interest)


    data_loc = sprintf('%s/appearance_data.mat', output_dir);
    
    % Check if the data has been created, if not the extract it
    if(~exist(data_loc, 'file'))
        
        fprintf('Extracting feature point locations\n')
        % where feature points are stored        
        fp_dir = [output_dir, '/feature_points/'];
        extract_landmarks(DISFA_dir, [output_dir, '/feature_points/']);
        fprintf('Feature points extracted\n')
        
        fprintf('Extracting appearance data\n')
        [data_appearance, data_geom, vid_id, vid_user] = extract_appearance_features(DISFA_dir, fp_dir, patch_size, scale, landmarks_of_interest);

        patches_around_landmarks = landmarks_of_interest;

        save(data_loc, 'data_appearance', 'data_geom', 'vid_id', 'patches_around_landmarks', 'scale', 'patch_size', 'vid_user');
    end
end