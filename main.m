% CSE 417A, Homework 6, Problem 1.
% Main function for calling the AdaBoost and plot the training error
% and testing error VS number of trees
% Author: Suichao Wu
% Date: 11/30/2014

clear all;
clc;
%format long;
%%
% Load data and filter the data about 1 and 3.
fprintf('Working on the one-vs-three problem...\n\n');
load zip.train;
fprintf('Filtering out the one and three training set...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y_tr = subsample(:,1);
X_tr = subsample(:,2:257);

load zip.test;
fprintf('Filtering out the one and three testing set...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y_te = subsample(:,1);
X_te = subsample(:,2:257);

n_trees = 100;
output15 = AdaBoost(X_tr, Y_tr, X_te, Y_te, n_trees);
train_error = output15(:,1);
test_error = output15(:,2);
tt = 1:1:n_trees;
% Plot the training error and testing error VS n_trees
figure
plot(tt, train_error, tt, test_error);
legend('training error', 'test error');
xlabel('Number of weak hypotheses');
ylabel('Error');
grid on;
title('Training error and test error for 1 VS 3');
%%
% Load data and filter the data about 3 and 5.
fprintf('\nNow working on the three-vs-five problem...\n\n');
fprintf('Filtering out the three and five training set...\n\n');
load zip.train
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_tr = subsample(:,1);
X_tr = subsample(:,2:257);

fprintf('Filtering out the three and five training set...\n\n');
load zip.test
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_te = subsample(:,1);
X_te = subsample(:,2:257);

n_trees = 200;
output35 = AdaBoost(X_tr, Y_tr, X_te, Y_te, n_trees);
train_error = output35(:,1);
test_error = output35(:,2);
tt = 1:1:n_trees;
% Plot the training error and testing error VS n_trees
figure
plot(tt, train_error, tt, test_error);
legend('training error', 'test error');
xlabel('Number of weak hypotheses');
ylabel('Error');
grid on;
title('Training error and test error for 3 VS 5');
