% This is a  1NN regressor with euclidean distance measure. 
% 
function KNN_prediction = KNN(label_instance,unlabel_instance, label)
    [idx, dist] = knnsearch(label_instance,unlabel_instance,'dist','euclidean','k',1);
    KNN_prediction = zeros(size(unlabel_instance,1),1);
    for t = 1 : size(unlabel_instance,1)
        KNN_prediction(t) = mean(label(idx(t,:)));
    end
end
