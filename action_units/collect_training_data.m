function [data_train, labels_train, data_valid, labels_valid, data_test, labels_test, dummy, sample_length, split] = ...
    collect_training_data(appearance_data, geom_data, landmarks_in_data, video_ids, landmarks_of_interest, au_train, rest_aus, users, train_users, test_users, tmp_data_dir)
dummy = 1;

%% This should be a separate function?

input_label_files = cell(numel(users),1);

% This is for loading the labels
for i=1:numel(users)
    
    if(exist('D:/Databases/DISFA', 'file'))
        root = 'D:/Databases/DISFA/';
    elseif(exist('F:/datasets/DISFA', 'file'))
        root = 'F:/datasets/DISFA/';
    else
       fprintf('Can not find the dataset\n'); 
    end
    
    input_label_files{i} = [root, '/ActionUnit_Labels/', users{i}, '/', users{i}];
end
out_dir = [tmp_data_dir, '/precomputed/'];
out_file = [out_dir, sprintf('%d_%d.mat', test_users(1), au_train)];

if(~exist(out_file, 'file'))
% Extracting the labels
labels_all = extract_au_labels(input_label_files, au_train);

% Getting the indices describing the splits (full version)
[training_inds, valid_inds, testing_inds, split] = construct_indices(video_ids, train_users, test_users);

% Getting the rebalanced training and validation indices and data
[inds_train_rebalanced, inds_valid_rebalanced, sample_length] = construct_balanced(labels_all, training_inds, valid_inds, rest_aus, input_label_files);

% can now extract the needed labels
labels_train = labels_all(inds_train_rebalanced);
labels_valid = labels_all(inds_valid_rebalanced);

% note that we do not balance or manipulate the test data
labels_test = labels_all(testing_inds);

%%

% NMF library
addpath('./nmf_bpas');

%% This is the dimensionality reduction section

% Normalising the appearance data
normed_appearance_data = extract_and_normalise_appearance_data(appearance_data, video_ids, landmarks_in_data, landmarks_of_interest);

% The rank of resulting factorisation

% ideally this should be cross-validated
rank = 24;

% as the patches are square
im_w = sqrt(size(appearance_data, 1));

% preallocate the train, validation and test data
data_train = zeros(rank*numel(landmarks_of_interest), numel(inds_train_rebalanced));
data_valid = zeros(rank*numel(landmarks_of_interest), numel(inds_valid_rebalanced));
data_test = zeros(rank*numel(landmarks_of_interest), sum(testing_inds));

% preallocate the basis matrices
Ws = zeros(rank, im_w^2, numel(landmarks_of_interest));

% The actual dimensionality reduction of appearance data
for i=1:numel(landmarks_of_interest)
    
    %% For dim reduction training use both training and validation sets
    raw_data_train = cat(2, normed_appearance_data(:,inds_train_rebalanced, i), normed_appearance_data(:,inds_valid_rebalanced, i));
    
    num_train = numel(inds_train_rebalanced);
    
    raw_data_test = normed_appearance_data(:, testing_inds, i);
    %%
    [W,H,iter,HIS]=nmf(raw_data_train, rank, 'type', 'sparse', 'tol', 1e-4, 'ALPHA', 0.5, 'BETA', 0.25);

    % use the coefficients inferred from nmf as training ones
    H_train = H;
    
    % project the test on basis using non-negative least squares
    H_test = zeros(rank, size(raw_data_test,2));
    for r=1:size(raw_data_test,2)
        H_test(:,r) = lsqnonneg(W, raw_data_test(:,r));
    end

    %%
    %     A_rec = W*H_train;
    %     err_rec2 = mean(abs(A_train - A_rec));
    % 
    %     A_rec = W*H_test;
    %     err_rec3 = mean(abs(A_test - A_rec));
    %
    %     to_vis = randperm(size(A_test,2));
    %
    %     to_vis = to_vis(1:20);
    %
    %     vis = zeros(numel(to_vis)*im_w, im_w * 3);
    %
    %     for k=1:numel(to_vis)
    %
    %         vis((k-1)*im_w+1:k*im_w,1:im_w) = reshape(A_test(:,to_vis(k)), 20, 20);
    %         vis((k-1)*im_w+1:k*im_w,im_w+1:im_w*2) = reshape(A_rec(:,to_vis(k)), 20, 20);
    %
    %         h3 = lsqnonneg(W, A_test(:,to_vis(k)));
    %
    %         vis((k-1)*im_w+1:k*im_w,im_w*2+1:im_w*3) = reshape(W * h3, 20, 20);
    %     end
    %
    %     figure;
    %     imagesc(vis);
    %     colormap('gray')
    %     axis equal
    % save the processed data so other experiments can use it

    %%
    % split back to validation and training sets
    data_train((i-1)*rank+1:i*rank,:) = H(:,1:num_train);
    data_valid((i-1)*rank+1:i*rank,:) = H(:,num_train+1:end);
    
    % unused TODO
    Ws(:,:,i) = W';
    
    %%
    data_test((i-1)*rank+1:i*rank,:) = H_test;

end


%%
geom_data_train = geom_data(:, inds_train_rebalanced);
geom_data_valid = geom_data(:, inds_valid_rebalanced);

geom_data_test = geom_data(:, testing_inds);

data_train = cat(1, data_train, geom_data_train);
data_valid = cat(1, data_valid, geom_data_valid);

data_test = cat(1, data_test, geom_data_test);

% normalise the data for the training
offsets = mean(cat(2, data_train, data_valid),2);
scaling = std(cat(2, data_train, data_valid)')';

data_train = bsxfun(@minus, data_train, offsets);
data_train = bsxfun(@times, data_train, 1./scaling);

data_valid = bsxfun(@minus, data_valid, offsets);
data_valid = bsxfun(@times, data_valid, 1./scaling);

data_test = bsxfun(@minus, data_test, offsets);
data_test = bsxfun(@times, data_test, 1./scaling);

if(~exist(out_dir, 'file'))
   mkdir(out_dir);
end

save(out_file, 'data_train', 'labels_train', 'data_valid', 'labels_valid', 'data_test', 'labels_test', 'dummy', 'sample_length', 'split');
else
   load( out_file);
end

end

function [training_inds, valid_inds, testing_inds, split] = construct_indices(video_inds, train_users, test_users)


    % extract these separately so as to guarantee person independent splits for
    % validation
    split = round(2*numel(train_users)/3);

    users_train = train_users(1:split);

    training_inds = false(size(video_inds));
    for i=1:numel(users_train)
        training_inds = training_inds | video_inds == users_train(i);
    end
    
    users_valid = train_users(split+1:end);
    valid_inds = false(size(video_inds));
    for i=1:numel(users_valid)
        valid_inds = valid_inds | video_inds == users_valid(i);
    end
        
    testing_inds = false(size(video_inds));
    for i=1:numel(test_users)
        testing_inds = testing_inds | video_inds == test_users(i);
    end
end

function [inds_train_rebalanced, inds_valid_rebalanced, sample_length] = construct_balanced(labels_all, training_inds, valid_inds,...
                                                                            rest_aus, input_labels_files)
    
    all_subs = 1:size(labels_all,1);
                                                                        
    labels_train = labels_all(training_inds);
    labels_valid = labels_all(valid_inds);
    
    labels_other_au = zeros(size(labels_all,1), numel(rest_aus));

    % This is used to pick up activity of other AUs for a more 'interesting'
    % data split and not only neutral expressions for negative samples    
    for i=1:numel(rest_aus)
        labels_other_au(:,i) = extract_au_labels(input_labels_files, rest_aus(i));
    end
    
    labels_other_au_train = labels_other_au(training_inds,:);
    labels_other_au_valid = labels_other_au(valid_inds,:);
    
    [ inds_train_rebalanced, sample_length ] = extract_events( labels_train, labels_other_au_train );
    [ inds_valid_rebalanced ] = extract_events( labels_valid, labels_other_au_valid, sample_length );
    
    % Need to move from indices in labels_train space to indices in
    % original labels_all space
    sub_train = all_subs(training_inds);
    inds_train_rebalanced = sub_train(inds_train_rebalanced);
    
    sub_valid = all_subs(valid_inds);
    inds_valid_rebalanced = sub_valid(inds_valid_rebalanced);
end