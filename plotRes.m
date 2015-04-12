% Author: Suichao Wu
% Date: 11/30/2014
train_error = output15(:,1);
test_error = output15(:,2);
n_trees = 100;
tt = 1:1:n_trees;
% Plot the training error and testing error VS n_trees
figure
subplot(2,1,1);
plot(tt, train_error, tt, test_error);
legend('training error', 'test error');
xlabel('Number of weak hypotheses');
ylabel('Error');
grid on;
title('Training error and test error for 1 VS 3');
n_trees = 200;
train_error = output35(:,1);
test_error = output35(:,2);
tt = 1:1:n_trees;
% Plot the training error and testing error VS n_trees
subplot(2,1,2);
plot(tt, train_error, tt, test_error);
legend('training error', 'test error');
xlabel('Number of weak hypotheses');
ylabel('Error');
grid on;
title('Training error and test error for 3 VS 5');