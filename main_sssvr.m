clear; clc;
final_average = zeros(3,9);

% load('chenting_20151124_noon_2.mat');  %label
% load('de_LDS_chenting.mat');  %instance
% final_average(1,:) =  sssvr(de_LDS,perclos);
% clear de_LDS;
% clear perclos;

% load('chuxing_20151106_noon.mat');  %label
% load('de_LDS_chuxing.mat');  %instance
% final_average(2,:) =  sssvr(de_LDS,perclos);
% clear de_LDS;
% clear perclos;
% 
load('duyuming_20151024_noon.mat');  %label
load('de_LDS_duyuming.mat');  %instance
final_average(3,:) =  sssvr(de_LDS,perclos);
clear de_LDS;
clear perclos;