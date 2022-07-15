function [sln Safer_prediction]= SAFER(candidate_prediction,baseline_prediction)

% SAFER implements the SAFER algorithm in [1].
%  ========================================================================
%
%  Input:
%  SAFER takes 2 input parameters in this order:
%
%  candidate_prediction: a matrix with size instance_num * candidate_num . Each column
%                        vector of candidate_prediction is a candidate regression result. 
%                        
%  baseline_prediction: a column vector with length instance_num. It is the regression result
%                       of the baseline method.
%
%  In our paper, all the feautures and labels are normalized to [0,1].
%
%  ========================================================================
%
%  Output:
%  Safer_prediction: a predictive regression result by SAFER.
%
%  ========================================================================
%
%  Example:
%    f = [f1 f2];
%    [Safer_prediction]=SAFER(f,f0);
%
%  ========================================================================
%
%  Reference:
%  [1]  Yu-Feng Li, Han-Wen Zha and Zhi-Hua Zhou. Construct Safe Prediction from Multiple Regressors. In: The 31st AAAI Conference on Artificial Intelligence %  (AAAI'17), San Francisco, California, 2017.
%
        candidate_num = size(candidate_prediction,2);

        H = candidate_prediction'*candidate_prediction*2;
        f = -2 * candidate_prediction' * baseline_prediction;

        Aeq = ones(1,candidate_num);
        beq = 1;

        lb = zeros(candidate_num,1);
        ub = ones(candidate_num,1);

        options=optimset('Algorithm', 'interior-point-convex','Display','off');

        % use quadprogramming to get the final result.
        [sln,fval] = quadprog(H,f,[],[],Aeq, beq, lb, ub,[],options);

        Safer_prediction = 0;
        for i = 1:candidate_num
            Safer_prediction = Safer_prediction + sln(i)*candidate_prediction(:,i);
        end
        
 end