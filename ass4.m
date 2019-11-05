table1 = readtable('/Users/harsha/Desktop/3-1/NNFL/assig 1/data.xlsx');

X0 = ones(size(table1,1),1); 
X = [table1(:,1:2)];

X= table2array(X);

% Inputs normalization 
X(:,2) = (X(:,2)-mean(X(:,2)))/std(X(:,2)); 
X(:,1) = (X(:,1)-mean(X(:,1)))/std(X(:,1)); 
X= [X0 X];

%Output vector normalization 
table1= table2array(table1);

y = (table1(:,3)-mean(table1(:,3)))/std(table1(:,3)); 
clear table1 
z=inv(X'*X);
w = z*X'*y; 
%Weight evaluation using vectorised linear regression 
w_gd = [0.0000; 0.0782; 0.3609]; 
%Weights from linear regression - batch gradient descent
w_sgd = [-0.0102; 0.0789; 0.3057];
%Weights from linear regression - stochastic gradient descent
e1 = sqrt(sum((w-w_gd).^2)); 
%Error with respect to batch gradient descent 
e2 = sqrt(sum((w-w_sgd).^2)); 
%Error with respect to stochastic gradient descent 
