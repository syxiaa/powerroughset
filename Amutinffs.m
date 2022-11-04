%% DEMO FILE
close all
clc;
% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% 读取一个文件夹下的所有csv文件
fileFolder=fullfile('F:\UCI1\');
dirOutput=dir(fullfile(fileFolder,'*.csv')); 
datasets={dirOutput.name};
m = size(datasets,2);
for i=1:m
    filename = strcat('F:\UCI1\',datasets(1,i));
    filename = filename{1};
    dataset = csvread(filename);
    M = size(dataset,1);  % 数据个数
    N = size(dataset,2);  % 特征个数
    X = dataset(:,2:N); % 第一列是标签
    Y = dataset(:,1);
    numF = size(X,2);
    

    %  有监督
    %  'mutinffs'
    [ ranking_mutinffs , w] = mutInfFS( X, Y, numF );
    savepath = strcat('F:\约简结果\mutinffs\',datasets(1,i));
    savepath = savepath{1};
    dlmwrite(savepath,ranking_mutinffs','delimiter', ',' , '-append');
     
end