function [ f1 ] = calculateF1(precision, recall)
	f1 = 2*(precision*recall) / (precision + recall);