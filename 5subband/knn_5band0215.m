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

load('chenting_20151124_noon_2.mat');  %label
load('de_LDS_chenting.mat');  %instance
de_LDS0 = [de_LDS(:,:,1);de_LDS(:,:,2);de_LDS(:,:,3);de_LDS(:,:,4);de_LDS(:,:,5)]; 
de_LDS1 = de_LDS(:,:,1);
de_LDS2 = de_LDS(:,:,2);
de_LDS3 = de_LDS(:,:,3);
de_LDS4 = de_LDS(:,:,4);
de_LDS5 = de_LDS(:,:,5);

de_LDS0 = de_LDS0';
de_LDS1 = de_LDS1';
de_LDS2 = de_LDS2';
de_LDS3 = de_LDS3';
de_LDS4 = de_LDS4';
de_LDS5 = de_LDS5';

ln = 100; 
% rmse_baseline_pre = 0;
% rmse_pre = 0;
safe_count = 0;
iter = 100;
sum_rmse = 0;
sum_R = 0;
sum0_rmse = 0;
sum0_R = 0;
sum_sln = [0;0;0;0;0];
for i = 1:iter
    [~,idx] = sort(rand(885,1));    % 随机选择885样本中ln个
    % 选择标记
    label = perclos(idx(1:ln),:);
    ground_truth = perclos(idx(ln+1:885),:);
    % f0
    label_instance = de_LDS0(idx(1:ln),:);
    unlabel_instance = de_LDS0(idx(ln+1:885),:);

    label_num = size(label_instance,1);
    unlabel_num = size(unlabel_instance,1);
    X = minmax_normalized([label_instance;unlabel_instance]);
    label_instance = X(1:label_num,:);
    unlabel_instance = X(label_num+1:end,:);
    
    y = minmax_normalized([label;ground_truth]);
    label = y(1:label_num);
    ground_truth = y(label_num+1:end,:);
    
    baseline_prediction = KNN(label_instance,unlabel_instance, label);
    
    % f1
    label_instance = de_LDS1(idx(1:ln),:);
    unlabel_instance = de_LDS1(idx(ln+1:885),:);
    
    label_num = size(label_instance,1);
    unlabel_num = size(unlabel_instance,1);
    X = minmax_normalized([label_instance;unlabel_instance]);
    label_instance = X(1:label_num,:);
    unlabel_instance = X(label_num+1:end,:);
    
    y = minmax_normalized([label;ground_truth]);
    label = y(1:label_num);
    ground_truth = y(label_num+1:end,:);
    
    prediction1 = Self_KNN(label_instance,unlabel_instance,label,'euclidean',3);
    
    % f2
    label_instance = de_LDS2(idx(1:ln),:);
    unlabel_instance = de_LDS2(idx(ln+1:885),:);
    
    label_num = size(label_instance,1);
    unlabel_num = size(unlabel_instance,1);
    X = minmax_normalized([label_instance;unlabel_instance]);
    label_instance = X(1:label_num,:);
    unlabel_instance = X(label_num+1:end,:);
    
    y = minmax_normalized([label;ground_truth]);
    label = y(1:label_num);
    ground_truth = y(label_num+1:end,:);
    
    prediction2 = Self_KNN(label_instance,unlabel_instance,label,'euclidean',3);
    
    % f3
    label_instance = de_LDS3(idx(1:ln),:);
    unlabel_instance = de_LDS3(idx(ln+1:885),:);
    
    label_num = size(label_instance,1);
    unlabel_num = size(unlabel_instance,1);
    X = minmax_normalized([label_instance;unlabel_instance]);
    label_instance = X(1:label_num,:);
    unlabel_instance = X(label_num+1:end,:);
    
    y = minmax_normalized([label;ground_truth]);
    label = y(1:label_num);
    ground_truth = y(label_num+1:end,:);
    
    prediction3 = Self_KNN(label_instance,unlabel_instance,label,'euclidean',3);
    
    % f4
    label_instance = de_LDS4(idx(1:ln),:);
    unlabel_instance = de_LDS4(idx(ln+1:885),:);
    
    label_num = size(label_instance,1);
    unlabel_num = size(unlabel_instance,1);
    X = minmax_normalized([label_instance;unlabel_instance]);
    label_instance = X(1:label_num,:);
    unlabel_instance = X(label_num+1:end,:);
    
    y = minmax_normalized([label;ground_truth]);
    label = y(1:label_num);
    ground_truth = y(label_num+1:end,:);
    
    prediction4 = Self_KNN(label_instance,unlabel_instance,label,'euclidean',3);
    
    % f5
    label_instance = de_LDS5(idx(1:ln),:);
    unlabel_instance = de_LDS5(idx(ln+1:885),:);
    
    label_num = size(label_instance,1);
    unlabel_num = size(unlabel_instance,1);
    X = minmax_normalized([label_instance;unlabel_instance]);
    label_instance = X(1:label_num,:);
    unlabel_instance = X(label_num+1:end,:);
    
    y = minmax_normalized([label;ground_truth]);
    label = y(1:label_num);
    ground_truth = y(label_num+1:end,:);
    
    prediction5 = Self_KNN(label_instance,unlabel_instance,label,'euclidean',3);
    
    
   % integration
    candidate_prediction = [prediction1 prediction2 prediction3 prediction4 prediction5];
    [sln Safer_prediction]= SAFER(candidate_prediction,baseline_prediction);
    
    %rmse
    mse0 = sum((baseline_prediction-ground_truth).^2)/unlabel_num;
    mse = sum((Safer_prediction-ground_truth).^2)/unlabel_num;
    
    rmse0 = sqrt(mse0);
    rmse = sqrt(mse);

    
    % safe count
    if rmse <= rmse0 
        safe_count = safe_count + 1;
    end
    safe_proportion = safe_count / iter;
    

     % corr
    R0 = corr(baseline_prediction,ground_truth,'type','Pearson');
    R = corr(Safer_prediction,ground_truth,'type','Pearson');
   
     
    
    % sum rmse/corr/sln
    sum_rmse = sum_rmse + rmse;
    sum0_rmse = sum0_rmse + rmse0;
    
    sum_R = sum_R + R; 
    sum0_R = sum0_R + R0; 
    
    sum_sln = sum_sln +sln;
%     sln
end
safe_proportion
% average rmse/corr/sln
mean_rmse = sum_rmse/iter
mean0_rmse = sum0_rmse/iter
mean_R = sum_R/iter
mean0_R = sum0_R/iter
mean_sln = sum_sln/iter

