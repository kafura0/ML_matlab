function y = myKNN(X, Y, x, k)
%PRE: X is m x n feature matrix (m ...
%   Detailed explanation goes here

    % -- Helper variables
    [m n] = size(X);

    % -- Distance between test point and all training examples:
    d = X-repmat(x, m, 1);
    dd = d.^2;
    D = sum(dd, 2);
    
    [D_s, I] = sort(D);
    
    Y_s = Y(I(1:k));

    y = round(sum(Y_s)/k);
    
    y = y;

end
