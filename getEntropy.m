% Author: Suichao Wu
% Date: 11/30/2014
function entropy = getEntropy(input)
% Calculate entropy
if input == 0
    entropy = 0;
else
    entropy = -(input*log2(input) + (1-input)*log2(1-input));
end
end