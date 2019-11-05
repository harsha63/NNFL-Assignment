clear all;
close all;
clc;
data = readtable('/Users/harsha/Desktop/3-1/NNFL/assig 1/data4.xlsx');
data= table2array(data);
data(:,1:7) = (data(:,1:7)-mean(data(:,1:7)))./std(data(:,1:7));
p = randperm(size(data,1));
train_data = [zeros(fix(0.6*size(data,1)),1) data(p(1:0.6*size(data,1)),:)]; 
test_data = [zeros(fix(0.4*size(data,1)),1) data(p(0.6*size(data,1)+1:end),:)]; 

k = 500; 
alpha = 0.0001;
t = train_data;
for i = 1:size(train_data,1)
 if train_data(i,9)==1
    t(i,9)=0;
 else
    t(i,9)=1;
 end
end
w1 = rand(1,size(t,2)-1); 
y = t(:,9); 
for i=1:k
 h = logsig(w1*t(:,1:8)')'; 
 for j = 1:size(t,2)-1
 w1(j) = w1(j)-alpha*sum((y.*(1-h)+(y-1).*h).*t(:,j));
 end
end
clear y h
t = train_data;
for i = 1:size(train_data,1)
 if train_data(i,9)==2
 t(i,9)=0;
 else
 t(i,9)=1;
 end
end
w2 = rand(1,size(t,2)-1); 
y = t(:,9); 
for i=1:k
 h = logsig(w2*t(:,1:8)')'; 
 for j = 1:size(t,2)-1
 w2(j) = w2(j)-alpha*sum((y.*(1-h)+(y-1).*h).*t(:,j));
 end
end
clear y h
t = train_data;
for i = 1:size(train_data,1)
 if train_data(i,9)==3
 t(i,9)=0;
 else
 t(i,9)=1;
 end
end
w3 = rand(1,size(t,2)-1); 
y = t(:,9); 
for i=1:k
 h = logsig(w3*t(:,1:8)')'; 
 for j = 1:size(t,2)-1
 w3(j) = w3(j)-alpha*sum((y.*(1-h)+(y-1).*h).*t(:,j));
 end
end
clear y h t i j
yt_1 = logsig(w1*test_data(:,1:8)')'; 
yt_2 = logsig(w2*test_data(:,1:8)')'; 
yt_3 = logsig(w3*test_data(:,1:8)')'; 
a = [yt_1 yt_2 yt_3];
yp = zeros(size(yt_1,1),1);
for i = 1:size(yt_1,1)
 [~, yp(i)] = max(a(i,:));
end
c = zeros(3); 
for i = 1:size(yp,1)
 if test_data(i,9) == 1
 c(1,yp(i)) = c(1,yp(i))+1;
 elseif test_data(i,9) == 2
 c(2,yp(i)) = c(2,yp(i))+1;
 else
 c(3,yp(i)) = c(3,yp(i))+1;
 end
end
individual = zeros(1,3); 
overall = 0; 
for i = 1:3
 individual(i) = c(i,i)/sum(c(i,:));
 overall = overall + c(i,i);
end
overall = overall/sum(c(:));
ConfusionMatrix = c; 
clear c i yt_1 yt_2 yt_3 a¡