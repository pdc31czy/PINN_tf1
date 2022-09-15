%%%%%%%%%%%%%%%%%%%
%author: Ziyan Chen
%%%%%%%%%%%%%%%%%%%

%正确的版本_Implicit Scheme
%save('FDMdata.mat','x','tt','uu')
clc;
clear all;
close all;

%discretization
a = 0; b = 1;
t = 0;
M = 511; %the number of mesh of space
N = 200;%the number of mesh of time
dx =(b-a)/M; %space step
dt = 0.005; % time step size
r = dt / (dx^2); %Convergence constant
x = a:dx:b;
ua = 0; %Left Boundary Condition
uExact = zeros(M+1,1);
uNew = zeros(M+1,1);
U = zeros(M,1);
A = sparse(M,M);
F = zeros(M,1);

for i = 1:M
    U(i) = sin(2*pi*x(i+1));%Initial Condition
end

uNew = [ua;U];

%plot(x,uNew);

tt=zeros(N+1,1);
tt(1)=0;

uu=zeros(M+1,N+1);
uu(:,1)=uNew;

for n = 1:N %time loop
    t = t+dt %next time level
    tt(n+1) = t;
    uxb = 2*pi*exp(-t);%Right Boundary derivative
       
    %Construct matrix A
    for i=1:M-1
        A(i,i)=1+2*r;
        A(i+1,i)=-r;
        A(i,i+1)=-r;
    end
    A(M,M)=1+2*r;
    A(M,M-1)=-2*r;

    %Construct matrix F
    %function f
    for i = 1:M
        f(i) = exp(-t).*sin(2.*pi.*x(i+1)).*(4.*pi.*pi-1);
    end
    %matrix F
    for i = 1:M
        F(i) = U(i) + f(i)*dt;
    end
    F(1)=F(1)+r*ua;
    F(M)= F(M)+(2*r*uxb*dx); %update last row of B
    U = A\F; %Solving the system to get unknowns
    uNew = [ua;U];

    uu(:,n+1)=uNew;
    
    %Exact Solution
    for i = 1:M+1
        uExact(i) = sin(2.*pi.*x(i)).*exp(-t);
    end
%{
    subplot(2,2,1),plot(x,uNew); set(gca,'Ylim',[-1,1]);
    title('implicit shecheme U');
    subplot(2,2,2),plot(x,uExact);set(gca,'Ylim',[-1,1]);
    title('Exact Solution U');
    subplot(2,2,3),plot(x,abs(uExact-uNew)); set(gca,'Ylim',[0,0.005]);
    title('Error');

    pause(0.5);
%}
end
tt = reshape(tt, 1, N+1);
