function Prepare_data_JANUS()

    % Read in in-the-wild data and Multi-PIE data and concatenate them
    root = '../prepared_data/';
    Collect_combined_data(root, '0.25');
    Collect_combined_data(root, '0.35');
    Collect_combined_data(root, '0.5');
        
end

function Collect_combined_data(root, scale)

    wild_locs = dir(sprintf('%s/wild_%s*.mat', root, scale));
    mpie_locs = dir(sprintf('%s/mpie_%s*.mat', root, scale));
    aflw_locs = dir(sprintf('%s/aflw_%s*.mat', root, scale));
    % For a particular loaded m_pie check if an appropriate in-the-wild
    % exists
    for i=1:numel(aflw_locs)
       
        load([root, mpie_locs(i).name]);
        
        centres_pie = centres;
        
        wild_to_use = -1;
        for j=1:numel(wild_locs)
            load([root, wild_locs(j).name], 'centres');
            if(isequal(centres_pie, centres))
                wild_to_use = j;
            end                
        end
        
        % Reset the centres
        centres = centres_pie;
        
        if(wild_to_use ~= -1)            
            samples_bef = all_images;
            used_bef = actual_imgs_used;
            locations_bef = landmark_locations;
            load([root, wild_locs(wild_to_use).name]);
            actual_imgs_used = cat(1, actual_imgs_used, used_bef);
            all_images = cat(1, all_images, samples_bef);
            landmark_locations = cat(1, landmark_locations, locations_bef);
        end
        
        aflw_to_use = -1;
        for j=1:numel(aflw_locs)
            load([root, aflw_locs(j).name], 'centres');
            if(isequal(centres_pie, centres))
                aflw_to_use = j;
            end                
        end
        
        % Reset the centres
        centres = centres_pie;
        
        if(aflw_to_use ~= -1)
            samples_bef = all_images;
            used_bef = actual_imgs_used;
            locations_bef = landmark_locations;
            load([root, aflw_locs(aflw_to_use).name]);
            actual_imgs_used = cat(1, actual_imgs_used, used_bef);
            all_images = cat(1, all_images, samples_bef);
            landmark_locations = cat(1, landmark_locations, locations_bef);
        end        
        
        % Need to do the visibilities properly by actually seeing how many
        % labeled landmarks are at a particular view
        visiIndex = (sum(sum(landmark_locations(:,:,1)~=0, 3),1) / size(all_images,1)) > 0.4;
        save(sprintf('%s/JANUS_%s_%d.mat', root, scale, i), 'actual_imgs_used', 'all_images', 'centres', 'landmark_locations', 'training_scale', 'visiIndex');
    end
    
end