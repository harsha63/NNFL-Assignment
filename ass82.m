X = readtable('/Users/harsha/Desktop/3-1/NNFL/assig 1/data4.xlsx');
X=table2array(X);
%Input normalization 
X(:,1:4) = (X(:,1:4)-mean(X(:,1:4)))./std(X(:,1:4)); 
p = randperm(size(X,1)); %Shuffling data 
%Dividing the data into training data (60%) and testing data (40%) 
train_data = [zeros(fix(0.6*size(X,1)),1) X(p(1:0.6*size(X,1)),:)]; 
test_data = [zeros(fix(0.4*size(X,1)),1) X(p(0.6*size(X,1)+1:end),:)];  
clear p X 
k = 500; %Number of iterations 
alpha = 0.0001; %Learning rate 
w1 = zeros(1,size(train_data,2)-1); %initial weights for 1 vs 2 
w2 = zeros(1,size(train_data,2)-1); %initial weights for 2 vs 3 
w3 = zeros(1,size(train_data,2)-1); %initial weights for 3 vs 1 
%1 vs 2 classifier 
t = []; 
for i = 1:size(train_data,1) 
if train_data(i,9)==2 
t = [t; train_data(i,1:8) 0]; 
elseif train_data(i,9)==1 
t = [t; train_data(i,1:8) 1]; 
end
end
y = t(:,9); %target outputs 
for i=1:k
g = logsig(w1*t(:,1:8)')'; end %hypothesis calculation 
%weight updates 
for j = 1:size(t,2)-1 
w1(j) = w1(j)-alpha*sum((y.*(1-g)+(y-1).*g).*t(:,j)); 
end
clear g end clear y t i j
%2 vs 3 classifier 
t = []; 
for i = 1:size(train_data,1) 
if train_data(i,9)==2 
t = [t; train_data(i,1:8) 0]; 
elseif train_data(i,9)==3 
t = [t; train_data(i,1:8) 1];
end
end
y = t(:,9); %target outputs
for i=1:k
g = logsig(w2*t(:,1:8)')'; %hypothesis calculation 
%weight updates
for j = 1:size(t,2)-1 
w2(j) = w2(j)-alpha*sum((y.*(1-g)+(y-1).*g).*t(:,j));
end
clear g 
end
clear y t i j
%3 vs 1 classifier 
t = []; 
for i = 1:size(train_data,1) 
if train_data(i,9)==3 
t = [t; train_data(i,1:8) 0]; 
elseif train_data(i,9)==1 
t = [t; train_data(i,1:8) 1]; 
end
end
y = t(:,9); %target outputs
for i=1:k 
g = logsig(w3*t(:,1:8)')'; %hypothesis calculation %weight updates
for j = 1:size(t,2)-1 
w3(j) = w3(j)-alpha*sum((y.*(1-g)+(y-1).*g).*t(:,j)); 
end
clear g 
end
clear y t i j
%testing with test data 
yp_1 = logsig(w1*test_data(:,1:8)')'; %prediction values of 1 vs 2
yp_2 = logsig(w2*test_data(:,1:8)')'; %prediction values of 2 vs 3 
yp_3 = logsig(w3*test_data(:,1:8)')'; %prediction values of 3 vs 1 
%converting prediction values of 1 vs 2 to classes 
yp1 = zeros(size(yp_1,1),1); 
for i = 1:size(yp_1,1) 
if yp_1(i)>0.5 
yp1(i) = 2; 
else 
yp1(i) = 1; 
end
end
%converting prediction values of 2 vs 3 to classes
yp2 = zeros(size(yp_2,1),1); 
for i = 1:size(yp_2,1) 
if yp_2(i)>0.5 
yp2(i) = 2;
else 
yp2(i) = 3; 
end
end
%converting prediction values of 3 vs 1 to classes 
yp3 = zeros(size(yp_3,1),1); 
for i = 1:size(yp_3,1) 
if yp_3(i)>0.5 
yp3(i) = 3; 
else 
yp3(i) = 1; 
end
end
%calculating the class predicted (mode of the 3 predicted classes) 
yp = zeros(size(yp_1,1),1); 
for i = 1:size(yp,1) 
if yp1(i)+yp2(i)+yp3(i)~=6 
yp(i) = mode([yp1(i) yp2(i) yp3(i)]); 
else %to resolve dispute if all classes are predicted the same number of times 
y_r = [yp_1(i) yp_2(i) yp_3(i)]; 
[~, yp(i)] = max(y_r); 
if yp(i)==1 
t=2; 
elseif yp(i)==2 
t=2; 
elseif yp(i)==3 
t=3; 
end
yp(i) = t; 
end
end
clear y_r end
c = zeros(3); %to calculate confusion matrix 
for i = 1:size(yp,1) 
if test_data(i,9) == 1 
c(1,yp(i)) = c(1,yp(i))+1; 
elseif test_data(i,9) == 2 
c(2,yp(i)) = c(2,yp(i))+1; 
else 
c(3,yp(i)) = c(3,yp(i))+1; 
end
end
IA = zeros(1,3); %individual accuracies of the 3 classifiers
OA = 0; %overall accuracy 
for i = 1:3
IA(i) = c(i,i)/sum(c(i,:)); 
OA = OA + c(i,i); 
end
OA = OA/sum(c(:));
ConfusionMatrix = c; %confusion matrix 
clear c i t 
