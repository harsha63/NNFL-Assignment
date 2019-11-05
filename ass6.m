X = readtable('/Users/harsha/Desktop/3-1/NNFL/assig 1/data2.xlsx');
X= table2array(X);

n = input('Enter number of iterations: '); 
%In this case, 100 
[v, c] = kMeansClustering(X,2,n);
%Calling the function for k-means clustering 
classColors = zeros(size(X,1), 3); 
%Array to assign class colors to each data element 
markerSizes = zeros(size(X,1), 1); 
%Array to assign marker size to each data element 
for row = 1 : size(X,1) 
if c(row)==1 
% Class 1 = blue 
classColors(row,:) = [0, 0, 1]; 
markerSizes(row) = 20;
else 
% Class 2 = red 
classColors(row, :) = [1, 0, 0]; 
markerSizes(row) = 20; 
end
end
clear row 
%Class vs feature values 1 
figure; scatter(X(:,1),c, markerSizes, classColors); hold on; scatter(v(:,1),1:2,40,[0 1 0],'LineWidth',2) 
%Class vs feature values 2 
figure; scatter(X(:,2),c, markerSizes, classColors); hold on; scatter(v(:,2),1:2,40,[0 1 0],'LineWidth',2) 
%Class vs feature values 3 
figure; scatter(X(:,3),c, markerSizes, classColors); hold on; scatter(v(:,3),1:2,40,[0 1 0],'LineWidth',2) 
%Class vs feature values 4 
figure; scatter(X(:,4),c, markerSizes, classColors); hold on; scatter(v(:,4),1:2,40,[0 1 0],'LineWidth',2)

