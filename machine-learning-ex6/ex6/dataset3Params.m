function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
% 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
% You need to return the following variables correctly.
C = 1;
sigma = 0.03;

C_i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_j = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

lc=size(C_i,1);
ls=size(sigma_j,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
best=10000;
for i=1:lc
    for j=1:ls
        model= svmTrain(X,y,C_i(i),@(x1,x2)gaussianKernel(x1,x2,sigma_j(j)));%这里的x1,x2使用方法还需要查一下
        predictions=svmPredict(model,Xval);
        prediction=mean(double(predictions ~= yval));
        if prediction<best;
           best=prediction;
           C=C_i(i);
           sigma=sigma_j(j);
        end
    end
end
 
 


%=====================================================

end
