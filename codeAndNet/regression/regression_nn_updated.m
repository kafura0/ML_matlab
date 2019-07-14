
clear;

% Loads in the appropiate data for this problem
%load('/data/assessment/binary smile/facialPoints.mat');
%load('/data/assessment/binary smile/labels.mat');
load('facialPoints.mat')
load('headpose.mat')


%Formats the data into the correct shape and format that the ANN is expecting
feat = reshape(points, 132, 8955);
feat = feat';

%Labels are the 6th column
labels = pose(:, 6);

%Combines labels and points and then shuffles the data
combined = [feat, labels];
shuffledArray = combined(randperm(size(combined, 1)), :);

%Splits the data back into input and output data
train_y = shuffledArray(:, 133);
train_x = shuffledArray;
train_x(:, 133) = [];

train_x = train_x';
train_y = train_y';

%Neural Network Configuration and initialisation
networkStructure = [15, 15, 5];
NET = newff(train_x, train_y, networkStructure);

NET.trainParam.mu = 1.0000e-03;
NET.trainFcn = 'trainlm';
NET.performFcn = 'mse';
NET.trainParam.max_fail = 8;

%10-cross validation
k = 10;
trainingSize = size(train_x);
performances = [k];

for i = 1:k
    %finds the current data split
    lower = ceil(trainingSize(:, 2) / 10.0) * (i - 1);
    upper = min(ceil(trainingSize(:, 2) / 10.0) + lower, trainingSize(:, 2));

    testDataX = [];
    testDataY = [];
    trainingDataX = train_x;
    trainingDataY = train_y;

    testDataX(:, 1) = train_x(:, 1);

    %Split the actual data appropiately with the current iteration
    counter = 0;
    for j = (lower + 1):upper
        testDataX(:, floor(j - lower)) = train_x(:, floor(j));
        testDataY(:, j - lower) = train_y(:, j);
        trainingDataX(:, j - counter) = [];
        trainingDataY(:, j - counter) = [];
        counter = counter + 1;
    end

    %Train the ANN
    [NET, TR] = train(NET, trainingDataX, trainingDataY);

    %Run the ANN with the split test data and calculate the performance of the ANN
    prediction = sim(NET, testDataX);

    performances(:, i) = perform(NET, testDataY, prediction);

end

%Calculate and output the performances and the rmse once the 10-cross
%validation has been completed
avg_performance = sum(performances) / size(performances, 2);
rmse = sqrt(performances);
avg_rmse = sum(rmse) / size(rmse, 2);

performances
avg_performance
rmse
avg_rmse
