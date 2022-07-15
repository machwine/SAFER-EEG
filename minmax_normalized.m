% realize the minmax normalize
function X = minmax_normalized(instance)
    low = min(instance,[],1);
    high = max(instance,[],1);
    X = instance - repmat(low,size(instance,1),1);
    X = X ./ repmat(high-low,size(instance,1),1);
end

