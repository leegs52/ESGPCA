clear all
clc

load ./data/synthetic/data.csv
X=data;
load ./data/VM/VMdata_group1.csv
X=VMdata_group1;

means = mean(X);
maxValues = max(X);
minValues = min(X);
X = 2 * (X - means) ./ (maxValues - minValues);

%synthetic data
%X_mean=mean(X,1);
%X=X-X_mean;

param.alpha = 0.9; %parameter1
param.lambda = 2; %parameter2

opts.k = 5; % the number of components
opts.group_num = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];
%5*ones(70,1)
%VM data 1-1: [6, 12, 90, 6, 20, 54, 18, 6, 12]
%VM data 1-2: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
%VM data 2: [38, 36, 38, 36, 38, 38]
opts.rho = 0.1;
opts.delta = 1.1;
opts.rho_max = 1e2;
opts.MAX_ITER = 100;
opts.epsilon = 0;

opts.QUIET=0;

[P,Z1,Z2,Q, history] = SGPCA(X, param, opts);

Sparse_P=P;
Sparse_P(Z2 == 0) = 0;
Sparse_P(Z1 == 0) = 0;
%Sparse_P is the loading matrix

colormap(flipud(gray)); % 흑백 색상 맵 설정
imagesc(abs(Sparse_P(:,1:opts.k)));
colorbar;