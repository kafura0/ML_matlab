%prefix
k = 10;

%Load data
load('facialPoints.mat');
load('labels.mat');

%Matrix transpose to meet nn toolbox requirements
points = reshape(points,132,150);
points = points.';
%Shuffle the dataset due to unwanted ordering in dataset
combined = [points labels];
shuffledarray = combined(randperm(size(combined,1)),:);
train_x = shuffledarray;
train_x(:,133) = [];
train_y = shuffledarray(:, 133);
train_x = train_x.';
train_y = train_y.';

%10-folds cross validationsni
%1.Data splitting
for i = 1:k
    if (i == 1)
        %when i = 1 the first 132*15 points along with
        %1*15 labels will be used for testing and remaining will be used
        %for training
        test_set = train_x(:,1:15);
        test_label = train_y(:,1:15);
    
        train_set = train_x(:,16:150);
        train_label = train_y(:,16:150);
        
        net = newff(train_set,train_label,[4]);
        net.trainParam.max_fail = 8;
        [net, TR] = train(net, train_set, train_label);
        t = sim(net, test_set);
        
        for z = 1:15
            if t(z) < 0.5
                rounded(z)=0;
            else
                rounded(z)=1;
            end
        end
        diff = rounded - test_label;
        hit = sum(diff == 0);
        accuracy(i) = hit/15;
        perf(i) = perform (net, test_label, t);
        
        confusionCalc(test_label, rounded, 2)
        
        f1(i) = (2*(accuracy(i)*1/15))/(accuracy(i)+1/15);
    elseif (i > 1 && i < k)  
        
        test_set = train_x(:,((i-1)*15+1):(i*15));
        test_label = train_y(:,((i-1)*15+1):(i*15));
        
        train_set = [train_x(:,1:((i-1)*15)) train_x(:,i*15+1)];
        train_label = [train_y(:,1:((i-1)*15)) train_y(:,i*15+1)];
        
        net = newff(train_set,train_label,[4]);
        net.trainParam.max_fail = 8;
        [net, TR] = train(net, train_set, train_label);
        t = sim(net, test_set);
        
        for z = 1:15
            if t(z) < 0.5
                rounded(z)=0;
            else
                rounded(z)=1;
            end
        end
        diff = rounded - test_label;
        hit = sum(diff == 0);
        accuracy(i) = hit/15;
        perf(i) = perform (net, test_label, t);
        
        confusionCalc(test_label, rounded, 2)
        f1(i) = (2*(accuracy(i)*1/15))/(accuracy(i)+1/15);
    else
        
        test_set = train_x(:,136:150);
        test_label = train_y(:,136:150);
        
        train_set = train_x(:,1:135);
        train_label = train_y(:,1:135);
        net = newff(train_set,train_label,[4]);
        net.trainParam.max_fail = 8;
        [net, TR] = train(net, train_set, train_label);
        t = sim(net, test_set);
        
        for z = 1:15
            if t(z) < 0.5
                rounded(z)=0;
            else
                rounded(z)=1;
            end
        end
        diff = rounded - test_label;
        hit = sum(diff == 0);
        accuracy(i) = hit/15;
        perf(i) = perform (net, test_label, t);
        confusionCalc(test_label, rounded, 2)
        
        f1(i) = (2*(accuracy(i)*1/15))/(accuracy(i)+1/15);
    end
end

%Print
accuracys = accuracy
performances = perf
avg_accuracy = sum(accuracy)/10
avg_performances = sum(perf)/10
avg_f1_score = sum(f1)/10

        
        
      
    