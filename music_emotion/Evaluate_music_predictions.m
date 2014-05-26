function [ correlations, rmss, long_corr, long_rms ] = Evaluate_music_predictions( test_labels, predictions )
%EVALUATE_MUSIC_PREDICTIONS Summary of this function goes here
%   Detailed explanation goes here

    sequence_length = 15;

    long_corr = corr(test_labels, predictions).^2;
    long_rms = sqrt(mean((test_labels - predictions).^2));

    num_seqs = numel(test_labels) / sequence_length;
    
    correlations = zeros(num_seqs, 1);
    rmss = zeros(num_seqs, 1);
    
    for i=1:num_seqs
        ind_begin = (i-1)*sequence_length + 1;
        ind_end = (i-1)*sequence_length + sequence_length;
        % correlations aren't squared, as otherwise we would lose the sign
        correlations(i) = corr(test_labels(ind_begin:ind_end), predictions(ind_begin:ind_end));
        rmss(i) = sqrt(mean((test_labels(ind_begin:ind_end) - predictions(ind_begin:ind_end)).^2));
    end
    
end

