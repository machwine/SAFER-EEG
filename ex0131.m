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

function [Sum] = ex0131(de_LDS,perclos)


de_LDS = [de_LDS(:,:,1);de_LDS(:,:,2);de_LDS(:,:,3);de_LDS(:,:,4);de_LDS(:,:,5)]; 
de_LDS = de_LDS';

ln = 60; 
% rmse_baseline_pre = 0;
% rmse_pre = 0;
safe_count = 0;
iter = 30;
sum_rmse = 0;
sum_R = 0;
sum0_rmse = 0;
sum0_R = 0;
for i = 1:iter
    [~,idx] = sort(rand(885,1));
    label_instance = de_LDS(idx(1:ln),:);
    unlabel_instance = de_LDS(idx(ln+1:885),:);
    label = perclos(idx(1:ln),:);
    ground_truth = perclos(idx(ln+1:885),:);
    
    % label = perclos(1:ln,:);
    % ground_truth = perclos(ln+1:885,:);
    % label_instance = de_LDS(:,1:ln);
    % unlabel_instance = de_LDS(:,ln+1:885);
    
    % label_instance = label_instance';
    % unlabel_instance = unlabel_instance';
    
    
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
    
    mse0 = sum((baseline_prediction-ground_truth).^2)/unlabel_num;
    mse = sum((Safer_prediction-ground_truth).^2)/unlabel_num;
    
   
    % Safer_prediction
    % ground_truth
     rmse0 = sqrt(mse0);
     rmse = sqrt(mse);

   
    sum_rmse = sum_rmse + rmse;
    sum0_rmse = sum0_rmse + rmse0;
    
%     P = [unlabel_instance',Safer_prediction'];
%     G = [unlabel_instance',ground_truth'];
   R = corr(Safer_prediction,ground_truth,'type','Pearson');
   sum_R = sum_R + R; 
   R0 = corr(baseline_prediction,ground_truth,'type','Pearson');
   sum0_R = sum0_R + R0; 
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   
end
rmse_mean = sum_rmse / iter;
R_mean = sum_R / iter;
rmse0_mean = sum0_rmse / iter;
R0_mean = sum0_R / iter;



% rmse_mean
% rmse0_mean
% R_mean
% R0_mean
end
