% This is a simple self-training kNN regressor. The algorithm takes 5 parameters. 
% 
% Input: 
% label_instance, unlabel_instance and label are explained at the begining.
% 
% distance_measure: a string to descrive the distance measure used in knnsearch, 
%                   such as 'euclidean', 'cosine', 'mahalanobis' etc.
%
% k: a number to decide the number of neighbours used in knnsearch.
%
%  Output:
%  Self_KNN_prediction: predictice regression result of the self-training
%  kNN regressor.

function Self_KNN_prediction = Self_KNN(label_instance, unlabel_instance, label, distance_measure, k)

    label_l = label;
	[idx, dist] = knnsearch(label_instance,unlabel_instance,'dist',distance_measure,'k',k);
    label_u = mean(label(idx),2);
    instance = [label_instance;unlabel_instance];
    label = [label_l;label_u];
    [idx, dist] = knnsearch(instance,unlabel_instance,'dist',distance_measure,'k',k);
    for i = 1 : 5
        label_u = mean(label(idx),2);
        label_last = label;
        label = [label_l;label_u];
    end
      Self_KNN_prediction = label_u;
end