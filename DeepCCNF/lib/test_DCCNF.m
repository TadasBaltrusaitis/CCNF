clear
clc

a=ones(4,6);
b=2*ones(6);
c=3*ones(6);
c(1,2)=10;
b(5,4)=11;
dul=ones(2,6)';
d={c,b,a};
%gradientCCNF([1,1,1,1]',[2,2,2,2],d,0,0,0,0,dul,0);



an=thetaInVector(d);
sizeTheta={};
sizeTheta{1}=[6,6];
sizeTheta{2}=[6,6];
sizeTheta{3}=[4,6];

guz=thetaInCell(an,sizeTheta);


