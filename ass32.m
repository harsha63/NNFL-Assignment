
table1 = readtable('/Users/harsha/Desktop/3-1/NNFL/assig 1/data.xlsx');
% X is the matrix containing feature vectors for all instances 
X0 = ones(size(table1,1),1); 
X = [table1(:,1:2)];

X= table2array(X);

% Inputs normalization 
X(:,2) = (X(:,2)-mean(X(:,2)))/std(X(:,2)); 
X(:,1) = (X(:,1)-mean(X(:,1)))/std(X(:,1)); 
X= [X0 X];
%Target outputs (normalized) 
table1= table2array(table1);

y = (table1(:,3)-mean(table1(:,3)))/std(table1(:,3)); 
clear table1 
w = [0 0 0]; %Initial weights 
alpha = 0.001; %Learning rate 
lambda = 0.25; %Regularisation parameter 
k = input('Enter number of iterations: '); %In this case, 350
m = 50; %Number of training instances taken in each iteration 
J = zeros(k*m+1,1); %Array to store cost for each iteration 
W = zeros(k*m+1,3); %Array to store weigthts for each iteration
J(1) = evaluatecostfunction2(X,y,w,lambda); %Calculating initial cost 
W(1,1) = w(1); 
W(1,2) = w(2); 
W(1,3) = w(3); 
for i=1:k 
a = randperm(size(X,1),m); %Shuffling the data for each iteration 
for j=1:m 
h = (X(a(j),:)*w')-y(a(j)); %Hypothesis calculation
for b=1:3 
w(b) = w(b) - alpha*(h'*X(a(j),b))-alpha*lambda*w(b); %Weight updates 
W((i-1)*m+j+1,b) = w(b); %Storing weights in array 
end
J((i-1)*m+j+1) = evaluatecostfunction2(X,y,w,lambda); 
%Calculating cost 
end
end
figure; plot(0:k*m,J) %Plotting cost vs number of iterations 
clear i j a b h
w1 = 0.4:-0.001:-0.2;
w2 = 0.8:-0.005:-0.2; 
J1 = zeros(length(w1),length(w2)); 
for i=1:length(w1) 
for j=1:length(w2) 
J1(i,j) = evaluatecostfunction2(X,y,[0 w1(i) w2(j)],lambda); 
end
end
%Stochastic gradient descent - cost vs w1 and w2 figure; 
figure; plot3(W(:,3),W(:,2),J,'color','r'); grid ON;
%3D plot 
figure; contour(w2,w1,J1); hold on; 
plot(W(:,3),W(:,2),'color','r') %Contour plot 
clear i j
