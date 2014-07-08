function [normalised_appearance_data] = extract_and_normalise_appearance_data(appearance_data, vid_ids, available_patches, landmarks_of_interest)
    
    im_h = sqrt(size(appearance_data,1));
    
    num_landmarks_of_interest = numel(landmarks_of_interest);
    
    normalised_appearance_data = zeros(im_h*im_h, size(appearance_data,2), num_landmarks_of_interest);
 
    % Map from the actual landmark to the one relative to the image
    patch_ind_im = zeros(size(landmarks_of_interest));
    for k=1:numel(landmarks_of_interest)
        patch_ind_im(k) = find(available_patches == landmarks_of_interest(k));
    end
    
    % This is done for per user normalisation
    users = unique(vid_ids);    
    
    for i = 1:numel(users)
       
        inds_curr_user = vid_ids == users(i);        
        
        % First collect all of the appearance vectors around the feature points
        % of interest        
        curr_user_patches = double(appearance_data(:, inds_curr_user, patch_ind_im));

        % now do person-specific mean normalisation of each patch
        for p=1:size(curr_user_patches, 3)
            
            curr_patch =  curr_user_patches(:,:,p)/255;
            
            m = mean(curr_patch,2);
            curr_patch = bsxfun(@minus, curr_patch, m) + 0.5;
            
            curr_patch(curr_patch < 0) = 0;
            curr_patch(curr_patch > 1) = 1;
            curr_user_patches(:,:,p) = curr_patch;
            
        end
        normalised_appearance_data(:,inds_curr_user,:) = curr_user_patches;
  
    end

end