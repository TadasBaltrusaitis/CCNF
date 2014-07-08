function [ all_inds, sample_length ] = extract_events( labels_train, labels_other_au, sample_length )
%EXTRACT_EVENTS Summary of this function goes here
%   Detailed explanation goes here

rng(0);
% first find the beginning and end of all events
events = labels_train > 0;
event_start_end = xor(events(1:end-1), events(2:end));
event_inds = find(event_start_end);
% make sure we finish on a full event
if(mod(numel(event_inds),2) == 1)
    event_inds = event_inds(1:end-1);
end

event_inds = reshape(event_inds, 2, numel(event_inds)/2)';

% find the lenght of AU events
event_length = event_inds(:,2) - event_inds(:,1);

% sample_length 
if(nargin < 3)
%     sample_length = 2 * round(mean(event_length));
    sample_length = round(1.5 * round(mean(event_length)));
    if(mod(sample_length,2) == 1)
        sample_length = sample_length + 1;
    end
end
% take all of the events and extract the sample_length surrounding region

events_mid_pos = round((event_inds(:,2) + event_inds(:,1))/2);

% randomise the start and end slightly to avoid overfitting
events_mid_pos = events_mid_pos + round(0.1*sample_length * randn(size(events_mid_pos)));

% remove overlapping events
dist_evs = abs(events_mid_pos(1:end-1)-events_mid_pos(2:end));
events_mid_pos = events_mid_pos(dist_evs > sample_length);

pos_events_to_extract = zeros(numel(events_mid_pos),2);
pos_events_to_extract(:,1) = events_mid_pos - sample_length/2;
pos_events_to_extract(:,2) = events_mid_pos + sample_length/2;
%%
pos_events = zeros(size(pos_events_to_extract,1), sample_length);
for i=1:size(pos_events_to_extract,1)
    
    pos_events(i,:) = pos_events_to_extract(i,1):pos_events_to_extract(i,2)-1;
    
end

pos_inds = pos_events';
pos_inds = pos_inds(:);

% make sure we don't go out of bounds
pos_inds(pos_inds > numel(labels_train)) = numel(labels_train);
pos_inds(pos_inds < 1) = 1;

%% create some negative events (where other AUs are activated, but not one
% of interest), especially where multiple events take place

% first find the beginning and end of all events
events = (sum(labels_other_au,2) > 6) & ~(labels_train > 0); 
event_start_end = xor(events(1:end-1), events(2:end));
event_inds = find(event_start_end);
% make sure we finish on a full event
if(mod(numel(event_inds),2) == 1)
    event_inds = event_inds(1:end-1);
end
event_inds = reshape(event_inds, 2, numel(event_inds)/2)';

% find the lenght of AU events
% event_length = event_inds(:,2) - event_inds(:,1);

% sample_length 
% sample_length = 2 * round(mean(event_length));

% take all of the events and extract the sample_length surrounding region

events_mid_neg = round((event_inds(:,2) + event_inds(:,1))/2);

% randomise the start and end slightly to avoid overfitting
events_mid_neg = events_mid_neg + round(0.1*sample_length * randn(size(events_mid_neg)));

% remove overlapping events
dist_evs = abs(events_mid_neg(1:end-1)-events_mid_neg(2:end));
events_mid_neg = events_mid_neg(dist_evs > sample_length);

neg_event_valid = true(numel(events_mid_neg),1);

for i=1:numel(events_mid_neg)
   
    if(any(abs(events_mid_pos - events_mid_neg(i)) < sample_length))
        neg_event_valid(i) = false;
    end
end
events_mid_neg = events_mid_neg(neg_event_valid);

neg_events_to_extract = zeros(numel(events_mid_neg),2);
neg_events_to_extract(:,1) = events_mid_neg - sample_length/2;
neg_events_to_extract(:,2) = events_mid_neg + sample_length/2;
%%
neg_events = zeros(size(neg_events_to_extract,1), sample_length);
for i=1:size(neg_events_to_extract,1)
    
    neg_events(i,:) = neg_events_to_extract(i,1):neg_events_to_extract(i,2)-1;
    
end

neg_inds = neg_events';
neg_inds = neg_inds(:);

%
all_inds = cat(1, pos_inds, neg_inds);
end

