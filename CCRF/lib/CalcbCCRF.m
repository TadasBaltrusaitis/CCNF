function b = CalcbCCRF( alpha, x, mask, useIndicators )
%CALCBPRF Summary of this function goes here
%   Detailed explanation goes here

%     b = zeros(size(x,1),1);
% 
%     for i=1:size(x,1)
%        b(i) = 2 *  x(i,:) * (alpha .* mask(i,:)'); 
%     end

    % vectorising above code
    if(useIndicators)
        b = 2 * (x .* mask)* alpha;
    else
        b = 2 * x * alpha;
    end
end

