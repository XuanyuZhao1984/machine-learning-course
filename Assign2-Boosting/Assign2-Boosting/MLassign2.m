%load training dataset, and extract X and y.
load breast-cancer;
X=full(data.X);
y=data.y;
number=size(X,1);
%initialize the distribution Dt as uniform distribution
Dt=ones(number,1)/number;
%set the precision of Dt to 8
digits(8);
Dt=vpa(Dt);
%require the user to input the number of iterations
prompt = 'How many iterations do you need ';
T = input(prompt);
alphat=zeros(T,1);%store weights for weak hypothesis ht
threshold=zeros(T,1);%store thresholds of ht
direction=zeros(T,1);%store directions of ht,direction of -1 -> 1 change
ind=zeros(T,1);%store which feature is chosen for ht

H=zeros(number,1); %weighted sum of weak hypotheses

for t=1:T,
    %stump represents weak hypothesis ht
    [stump] = build_stump(X,y,Dt);
    errort=stump.werr;% weighted error
    alphat(t)=0.5*log((1-errort)/errort);
    threshold(t)=stump.x0;
    direction(t)=stump.s;
    ind(t)=stump.ind;
    
    %update Dt
    ht_value=sign(direction(t)*(X(:,ind(t))-threshold(t)));
    %ignore normalization since it's done in "build_stump"
    Dt=Dt.*exp(-alphat(t)*y.*ht_value);
    %sum up ht_values with weights
    H=H+alphat(t)*ht_value;
    
    
end;

%take sign of the sum up values to mapping to {-1,+1}
H=sign(H);

%calculate error rate
error_Ada=sum(H~=y)/size(y,1);




%Part2, use SVM to classify the data and compare results in report
%SVM part is done in testSVM.m
%First we run this AdaBoost program, then run testSVM.m