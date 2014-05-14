function Write_patch_experts_ccnf(location_txt, location_mlab, trainingScale, centers, visiIndex, patch_experts, normalisationOptions, w_sizes)
      
    patches_file = fopen(location_txt, 'w');        
    
    [n_views, n_landmarks, ~] = size(patch_experts.correlations);

    fprintf(patches_file, '# scaling factor of training\r\n%f\r\n', trainingScale);
    
    % write out the scaling factor as this is what will be used when
    % fitting on the window
    fprintf(patches_file, '# number of views\r\n%d\r\n', n_views);

    % Write out the information about the view's and centers here
    fprintf(patches_file, '# centers of the views\r\n');

    for i=1:n_views
        % this indicates that we're writing a 3x1 double matrix
        writeMatrix(patches_file, centers(i,:)', 6);
    end
            
    fprintf(patches_file, '# visibility indices per view\r\n');
    
    for i=1:n_views
        % this indicates that we're writing a 3x1 double matrix
        writeMatrix(patches_file, visiIndex(i,:)', 4);
    end
    
    fprintf(patches_file, '# Sigma component matrices being used in these patches\r\n');    

    fprintf(patches_file, '# Number of windows sizes\r\n');
    fprintf(patches_file, '%d\r\n', numel(w_sizes));
    for w = 1:numel(w_sizes)
        fprintf(patches_file, '# Size of window\r\n');
        fprintf(patches_file, '%d\r\n', w_sizes(w));
        similarities = {};
        response_side_length = w_sizes(w);

        for st=1:size(patch_experts.patch_experts{1,1}.similarity_types, 1)
            type_sim = patch_experts.patch_experts{1,1}.similarity_types{st};
            neighFn = @(x) similarity_neighbor_grid(x, response_side_length, type_sim);
            similarities = cat(1, similarities, {neighFn});
        end

        sparsities = {};

        for st=1:size(patch_experts.patch_experts{1,1}.sparsity_types, 1)
            spFn = @(x) sparsity_grid(x, response_side_length, patch_experts.patch_experts{1,1}.sparsity_types(st,1), patch_experts.patch_experts{1,1}.sparsity_types(st,2));
            sparsities = cat(1, sparsities, {spFn});
        end
                
        region_length = response_side_length^2;

        % Adding the sparsities here if needed
        [ ~, PrecalcQ2s, ~, ~ ] = CalculateSimilarities( 1, {zeros(region_length,1)}, similarities, sparsities);

        PrecalcQ2s = PrecalcQ2s{1};
        fprintf(patches_file, '# Number of Sigma components\r\n');
        fprintf(patches_file, '%d\r\n', numel(PrecalcQ2s));
        for q2=1:numel(PrecalcQ2s)
            writeMatrix(patches_file, PrecalcQ2s{q2}, 5);
        end
    end    
    
    fprintf(patches_file, '# Patches themselves (1 line patches of a vertex)\r\n');
    
    for i=1:n_views
        for j=1:n_landmarks

            if(~visiIndex(i,j))                
                fprintf(patches_file, '%d %d %d %d ', 5, 11, 11, 0);
                fprintf(patches_file, '%d\r\n', 0);
            else
            num_neurons = size(patch_experts.patch_experts{i,j}.thetas, 1);
            
            % CCNF patch(5), width, height, num_neurons, Patch(2), neuron_type,
            % normalisation, bias, alpha, rows, cols, type            
            num_modalities = size(patch_experts.patch_experts{i,j}.thetas, 3);

            fprintf(patches_file, '%d %d %d %d ', 5, 11, 11, num_neurons);
            
            for n=1:num_neurons
                for m=1:num_modalities

                    if(strcmp(patch_experts.types{m}, 'reg'))
                       type = 0; 
                    elseif(strcmp(patch_experts.types{m}, 'grad'))
                       type = 1; 
                    else
                       fprintf('Not supported patch type\n');
                       type = 0;
                    end

                    % normalise the w
                    w = patch_experts.patch_experts{i,j}.thetas(n, 2:end, m);
                    norm_w = norm(w);
                    w = w/norm(w);
                    bias = patch_experts.patch_experts{i,j}.thetas(n, 1, m);
                    alpha = patch_experts.patch_experts{i,j}.alphas((m-1)*num_neurons+n);
                    % also add patch confidence based on correlation scores
                    fprintf(patches_file, '%d %d %f %f %f %d %d %d ', 2, type, norm_w, bias, alpha, 11, 11, 5);

                    % the actual weight matrix
                    fprintf(patches_file, '%f ', w);
                    fprintf(patches_file,'\r\n');
                end
            end
            
            % Write out the betas
            for b=1:numel(patch_experts.patch_experts{i,j}.betas)
                fprintf(patches_file, '%f ', patch_experts.patch_experts{i,j}.betas(b));
            end
            
            % finally write out the confidence
            fprintf(patches_file, '%f\r\n', patch_experts.correlations(i,j));
        end    
    end

    end

    fclose(patches_file);
    save(location_mlab, 'patch_experts', 'trainingScale', 'centers', 'visiIndex', 'normalisationOptions');
    