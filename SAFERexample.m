function example()

% This is an example to use SAFER method on UCI data set 'housing'. In particular, 10
% instances are labeled while the rest are unlabeled. 
% This example uses two self-training kNN regressors with different distance measures 
% as candidate regressors and a 1NN as baseline regressor.
%
% Some of the important variables are explained below:
%
% label_instance: a matrix with size label_instance_num * dimension. Each
%                 row vector of label_instance is a instance vector, of which
%                 the label is already known.
% unlabel_instance: a matrix with size unlabel_instance_num * dimension. Each
%                   row vector of unlabel_instance is a instance vector,
%                   of which the label is still unknown.
% label: a column vector with length label_instance_num. Each element
%        is a real number and the jth element is the label of the jth row
%        vector of label_instance. 
% ground_truth: a column vector with length unlabel_instance_num. Each element
%        is a real number and the jth element is the label of the jth row

%        vector of unlabel_instance. 
% 
% In our AAAI'17 experiment, all the features and labels are nomalized to [0,1] in advanced.
%

clear; clc;

label = [];ground_truth = []; label_instance = []; unlabel_instance = [];
load('housing10.mat');

label_num = size(label_instance,1);
unlabel_num = size(unlabel_instance,1);
X = minmax_normalized([label_instance;unlabel_instance]);
label_instance = X(1:label_num,:);
unlabel_instance = X(label_num+1:end,:);

y = minmax_normalized([label;ground_truth]);
label = y(1:label_num);
ground_truth = y(label_num+1:end,:);


prediction1 = Self_KNN(label_instance,unlabel_instance,label,'euclidean',3);
prediction2 = Self_KNN(label_instance,unlabel_instance,label,'cosine',3);
candidate_prediction = [prediction1 prediction2];

baseline_prediction = KNN(label_instance,unlabel_instance, label);

[Safer_prediction]= SAFER(candidate_prediction,baseline_prediction);


mse = sum((Safer_prediction-ground_truth).^2)/(label_num+unlabel_num);
mse0 = sum((baseline_prediction-ground_truth).^2)/(label_num+unlabel_num);
% Safer_prediction
% ground_truth
mse
mse0


% This is a  1NN regressor with euclidean distance measure. 
% 
function KNN_prediction = KNN(label_instance,unlabel_instance, label)
    [idx, dist] = knnsearch(label_instance,unlabel_instance,'dist','euclidean','k',1);
    KNN_prediction = zeros(size(unlabel_instance,1),1);
    for t = 1 : size(unlabel_instance,1)
        KNN_prediction(t) = mean(label(idx(t,:)));
    end
end

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

% realize the minmax normalize
function X = minmax_normalized(instance)
    low = min(instance,[],1);
    high = max(instance,[],1);
    X = instance - repmat(low,size(instance,1),1);
    X = X ./ repmat(high-low,size(instance,1),1);
end

end