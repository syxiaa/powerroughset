%% DEMO FILE
close all
clc;
% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

% ��ȡһ���ļ����µ�����csv�ļ�
fileFolder=fullfile('F:\UCI1\');
dirOutput=dir(fullfile(fileFolder,'*.csv')); 
datasets={dirOutput.name};
m = size(datasets,2);
for i=1:m
    filename = strcat('F:\UCI1\',datasets(1,i));
    filename = filename{1};
    dataset = csvread(filename);
    M = size(dataset,1);  % ���ݸ���
    N = size(dataset,2);  % ��������
    X = dataset(:,2:N); % ��һ���Ǳ�ǩ
    Y = dataset(:,1);
    numF = size(X,2);
    

    %  �мල
    %  'mutinffs'
    [ ranking_mutinffs , w] = mutInfFS( X, Y, numF );
    savepath = strcat('F:\Լ����\mutinffs\',datasets(1,i));
    savepath = savepath{1};
    dlmwrite(savepath,ranking_mutinffs','delimiter', ',' , '-append');
     
end