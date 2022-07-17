function [SUM] = sssvr(de_LDS,perclos)
% clear; clc;

% load('jiangxinglin_20151012_night.mat');  %label
% load('de_LDS_jiangxinglin151012night.mat');  %instance
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

SUM = zeros(1,9);
% rmse_baseline_pre = 0;
% rmse_pre = 0;

    safe_count = 0;
    iter = 100;
    sum_rmse = 0;
    sum_R = 0;
    sum0_rmse = 0;
    sum0_R = 0;
    sum_sln = [0;0;0;0;0];
    for i = 1:1
        % f0
        baseline_prediction = DemoSemiSVR(de_LDS0,perclos);
        % prediction0 = Self_KNN(label_instance,unlabel_instance,label,'euclidean',3);
        
        % f1
        prediction1 = DemoSemiSVR(de_LDS1,perclos);
        
        % f2
        prediction2 = DemoSemiSVR(de_LDS2,perclos);
        
        % f3
        prediction3 = DemoSemiSVR(de_LDS3,perclos);
        
        % f4
        prediction4 = DemoSemiSVR(de_LDS4,perclos);
        
        % f5
        prediction5 = DemoSemiSVR(de_LDS5,perclos);
        
        
        % integration
        candidate_prediction = [prediction1 prediction2 prediction3 prediction4 prediction5];
        [sln Safer_prediction]= SAFER(candidate_prediction,baseline_prediction);
        
        %rmse
        mse0 = sum((baseline_prediction-perclos).^2)/885;
        mse = sum((Safer_prediction-perclos).^2)/885;
        
        rmse0 = sqrt(mse0);
        rmse = sqrt(mse);
        
        
%         % safe count
%         if rmse <= rmse0
%             safe_count = safe_count + 1;
%         end
%         safe_proportion = safe_count / iter;
        
        
        % corr
        R0 = corr(baseline_prediction,perclos,'type','Pearson');
        R = corr(Safer_prediction,perclos,'type','Pearson');
        
        
        % sum rmse/corr/sln
%         sum_rmse = sum_rmse + rmse;
%         sum0_rmse = sum0_rmse + rmse0;
%         
%         sum_R = sum_R + R;
%         sum0_R = sum0_R + R0;
%         
%         sum_sln = sum_sln +sln;
        %     sln
    end
    
    % average rmse/corr/sln
%     mean_rmse = sum_rmse/iter;
%     mean0_rmse = sum0_rmse/iter;
%     mean_R = sum_R/iter;
%     mean0_R = sum0_R/iter;
%     mean_sln = sum_sln/iter;
    
    % select 5 label_num and get the average values
    SUM(1) = rmse;
    SUM(2) = rmse0;
    SUM(3) = R;
    SUM(4) = R0;
    for j = 1:5
        SUM(j+4) = sln(j);
    end
   
SUM
end