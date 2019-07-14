function [ confusion_matrix ] = confusionCalc( groundTruth, prediction, numClasses)
%CONFUSION Summary of this function goes here
%   Detailed explanation goes here
    if(size(groundTruth) ~= size(prediction))
        error("Missmatching matrix sizes");
    end
	
    confusion_matrix = zeros(numClasses, numClasses);
    
    for x = 1:size(prediction, 2)
        targetClass = groundTruth(:, x) + 1;
	    outputClass = prediction(:, x) + 1;
		
	    confusion_matrix(targetClass, outputClass) = confusion_matrix(targetClass, outputClass) + 1;
    end
end