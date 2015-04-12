% Author: Suichao Wu
% Date: 11/30/2014

function y = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use
[num_tr, num_f] = size(X_tr);
[num_te, num_f] = size(X_te);
% Initialize the weight vector for each example equally.
D = zeros(n_trees+1, num_tr); % Each row corresponds to one weak learner.
D(1,:) = 1 / num_tr; % The initial weight for each example is equal.
% Pre-rellocation
stumps = cell(n_trees, 1);
% Find all different values for each feature
val_feature = cell(num_f, 1);
for ii = 1:1:num_f
    val_feature{ii} = unique(X_tr(:,ii));
end
% Find the character we want to distinguish
y_label = unique(y_tr);
goal = y_label(1);
% Declare a cell array to store the info gains
infoGain = cell(n_trees, num_f);
ind_final = cell(n_trees, 1);
val_final = zeros(n_trees, 1);
leftLabel_final = zeros(n_trees, 1);
rightLabel_final = zeros(n_trees, 1);

yPred = zeros(num_tr, n_trees);
yTest = zeros(num_te, n_trees);
epsilon = zeros(n_trees, 1);
alpha = zeros(n_trees, 1);
Z = zeros(n_trees, 1);
sumH = zeros(num_tr, n_trees);
H = zeros(num_tr, n_trees);
yH = zeros(num_tr, n_trees);
num_err_train = zeros(n_trees, 1);
train_err = zeros(n_trees, 1);

num_err_test = zeros(n_trees, 1);
test_err = zeros(n_trees, 1);
yHTest = zeros(num_te, n_trees);
HTest = zeros(num_te, n_trees);
sumHTest = zeros(num_te, n_trees);

for t=1:1:n_trees
    % Train the weak learner by information gain based on D(t,:).
    IG_best = zeros(num_f, 1);
    val_best = zeros(num_f, 1);
    leftLabel_best = zeros(num_f, 1);
    rightLabel_best = zeros(num_f, 1);
   
    for ii = 1:1:num_f
        val_f = val_feature{ii}; % Different values for feature_ii.
        % Information gain vector for this feature for each value
        % Calculate the parent entropy
        indGoal = find(y_tr == goal);
        parent_entropy = getEntropy(sum(D(t,indGoal)));
        % loop through all feature values
        [val_f_row, val_f_col] = size(val_f);
        IG = zeros(val_f_row, 1); % store the Info gain for each value of feature ii
        leftLabel = zeros(val_f_row, 1);
        rightLabel = zeros(val_f_row, 1);
        for jj = 1:1:val_f_row
            indLeft = find(X_tr(:,ii) <= val_f(jj)); % left children's index
            indRight = find(X_tr(:,ii) > val_f(jj)); % right children's index
            % Calculate the left child entropy
            leftWgtTotal = sum(D(t,indLeft));
            indGoalLeft = indLeft(find(y_tr(indLeft) == goal));
            wgtGoal = sum(D(t,indGoalLeft));
            wgtOther = leftWgtTotal - wgtGoal;
            left_entropy = getEntropy(wgtGoal);
            if wgtGoal >= wgtOther
                leftLabel(jj) = goal;
            else
                leftLabel(jj) = y_label(2);
            end
            % Calculate the right child entropy
            rightWgtTotal = sum(D(t,indRight));
            indGoalRight = indRight(find(y_tr(indRight) == goal));
            wgtGoal = sum(D(t,indGoalRight));
            wgtOther = rightWgtTotal - wgtGoal;
            right_entropy = getEntropy(wgtGoal);
            if wgtGoal >= wgtOther
                rightLabel(jj) = goal;
            else
                rightLabel(jj) = y_label(2);
            end
            % Calcuate the information gain for value jj of feature ii.
            IG(jj, 1) = abs(parent_entropy - (leftWgtTotal*left_entropy + rightWgtTotal*right_entropy));
        end
        infoGain{t, ii} = IG;
        ind_best = find(IG == max(IG));
        IG_best(ii) = IG(ind_best(1));
        val_best(ii) = val_f(ind_best(1));
        leftLabel_best(ii) = leftLabel(ind_best(1));
        rightLabel_best(ii) = rightLabel(ind_best(1));
    end
    % Decide the stump with the most information gain
    ind_final{t,1} = find(IG_best == max(IG_best));
    bestInd = ind_final{t,1}(1);
    val_final(t) = val_best(bestInd);
    leftLabel_final(t) = leftLabel_best(bestInd);
    rightLabel_final(t) = rightLabel_best(bestInd);
    % Predict the y label for all examples using the stump
    yPred(find(X_tr(:,bestInd) <= val_final(t)),t) = leftLabel_final(t);
    yPred(find(X_tr(:,bestInd) > val_final(t)),t) = rightLabel_final(t);
    yTest(find(X_te(:,bestInd) <= val_final(t)),t) = leftLabel_final(t);
    yTest(find(X_te(:,bestInd) > val_final(t)),t) = rightLabel_final(t);
    % Measure "goodness" of this stump
    ind_err = find(yPred(:,t) ~= y_tr);
    kk = 1:1:num_tr;
    ind_all = kk';
    ind_corr = setdiff(ind_all, ind_err);
    epsilon(t) = sum(D(t,ind_err));
    alpha(t) = 0.5*log((1-epsilon(t))/epsilon(t));
    % Update the weight vector for the next stump
    D(t+1,ind_err) = D(t,ind_err)*exp(alpha(t));
    D(t+1,ind_corr) = D(t,ind_corr)*exp(-alpha(t));
    % Normalize D(t+1,:) by Z(t)
    Z(t) = sum(D(t+1,:));
    D(t+1,:) = D(t+1,:) / Z(t);
    
    % Calculate the traing error of the t combined stumps
    h = zeros(num_tr, t);
    for nn = 1:1:t
        h(find(yPred(:,nn) == y_label(1)),nn) = 1;
        h(find(yPred(:,nn) == y_label(2)),nn) = -1;
        sumH(:,t) = sumH(:,t) + alpha(nn)*h(:,nn);
    end
    H(:,t) = sign(sumH(:,t));
    yH(find(H(:,t) == 1), t) = y_label(1);
    yH(find(H(:,t) == -1), t) = y_label(2);
    [num_err_train(t,1),~] = size(find(yH(:,t) ~= y_tr));
    train_err(t, 1) = num_err_train(t,1) / num_tr;
    % Calculate the testing error of the t combined stumps
    h = zeros(num_te, t);
    for nn = 1:1:t
        h(find(yTest(:,nn) == y_label(1)),nn) = 1;
        h(find(yTest(:,nn) == y_label(2)),nn) = -1;
        sumHTest(:,t) = sumHTest(:,t) + alpha(nn)*h(:,nn);
    end
    HTest(:,t) = sign(sumHTest(:,t));
    yHTest(find(HTest(:,t) == 1), t) = y_label(1);
    yHTest(find(HTest(:,t) == -1), t) = y_label(2);
    [num_err_test(t,1),~] = size(find(yHTest(:,t) ~= y_te));
    test_err(t, 1) = num_err_test(t,1) / num_te;
end
y = [train_err test_err];
end
