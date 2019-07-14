%load('/data/assessment/binary smile/facialPoints.mat');
%load('/data/assessment/binary smile/labels.mat');
clear;

load('facialPoints.mat')
load('headpose.mat')


%format
feat = reshape(points, 132, 8955);
feat = feat';

labels = pose(:, 6);

combined = [feat, labels];
shuffledArray = combined(randperm(size(combined, 1)), :);

train_y = shuffledArray(:, 133);
train_x = shuffledArray;
train_x(:, 133) = [];

train_x = train_x';
train_y = train_y';

%Neural Network Stuff
networkStructure = [15, 15, 5];
NET = newff(train_x, train_y, networkStructure);

NET.trainParam.mu = 1.0000e-03; 
NET.trainFnc = 'trainlm';
NET.performFnc = 'mse';

%10 cross validation
k = 10;
trainingSize = size(train_x);
performances = [k];

for i = 1:k
    lower = ceil(trainingSize(:, 2) / 10.0) * (i - 1);
    upper = min(ceil(trainingSize(:, 2) / 10.0) + lower, trainingSize(:, 2));

    testDataX = [];
    testDataY = [];
    trainingDataX = train_x;
    trainingDataY = train_y;

    testDataX(:, 1) = train_x(:, 1);
    
    counter = 0;
    for j = (lower + 1):upper
        testDataX(:, floor(j - lower)) = train_x(:, floor(j));
        testDataY(:, j - lower) = train_y(:, j);
        trainingDataX(:, j - counter) = [];
        trainingDataY(:, j - counter) = [];
        counter = counter + 1;
    end

    [NET, TR] = train(NET, trainingDataX, trainingDataY);

    prediction = sim(NET, testDataX);
    
    performances(:, i) = perform(NET, testDataY, prediction);

end

avg_performance = sum(performances) / size(performances, 2);
rmse = sqrt(performances);
avg_rmse = sum(rmse) / size(rmse, 2);

performances
avg_performance
rmse
avg_rmse


