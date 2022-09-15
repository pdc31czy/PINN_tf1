%正确的版本Crank_Nicolson
%save('CNdata.mat','x','tt','uu','uuExact')
clc;
clear all;
close all;
tic;
%discretization
a = 0; b = 1;
t = 0;
c = 0; d = 0.5;
M = 255; %the number of mesh of space
N = 200;%the number of mesh of time %pinn200
dx =(b-a)/M; %space step
dt = (d-c)/N; % time step size %pinn0.005
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

%Exact Solution for t = 0
for i = 1:M+1
    uExact(i) = sin(2.*pi.*x(i)).*exp(-t);
end
uuExact = zeros(M+1,N+1);
uuExact(:,1)=uExact;

for n = 1:N %time loop
    t = t+dt %next time level
    tt(n+1) = t;
    uxb = 2*pi*exp(-t);%Right Boundary derivative
       
    %Construct matrix A
    for i=1:M-1
        A(i,i)=2+2*r;
        A(i+1,i)=-r;
        A(i,i+1)=-r;
    end
    A(M,M)=2+2*r;
    A(M,M-1)=-2*r;

    %Construct matrix F
    %function f
    for i = 1:M
        f(i) = exp(-t).*sin(2.*pi.*x(i+1)).*(4.*pi.*pi-1); %n+1层的f
    end
    for i = 1:M
        fn(i) = exp(-tt(n)).*sin(2.*pi.*x(i+1)).*(4.*pi.*pi-1); %n层的f
    end
    
    %matrix F
    for i = 2:M-1
        F(i) =r*U(i-1)+(2-2*r)*U(i)+r*U(i+1) +(f(i)+fn(i))*dt; 
    end
    F(1)=r*ua+(2-2*r)*U(1)+r*U(2)+(f(1)+fn(1))*dt+r*ua;
    F(M)=2*r*U(M-1)+(2-2*r)*U(M)+(f(M)+fn(M))*dt+(4*r*uxb*dx); 
    U = A\F; %Solving the system to get unknowns
    uNew = [ua;U];

    uu(:,n+1)=uNew;
    
    %Exact Solution
    for i = 1:M+1
        uExact(i) = sin(2.*pi.*x(i)).*exp(-t);
    end
    uuExact(:,n+1)=uExact;

    subplot(2,2,1),plot(x,uNew); set(gca,'Ylim',[-1,1]);
    title('implicit shecheme U');
    subplot(2,2,2),plot(x,uExact);set(gca,'Ylim',[-1,1]);
    title('Exact Solution U');
    subplot(2,2,3),plot(x,abs(uExact-uNew)); set(gca,'Ylim',[0,0.005]);
    title('Error');

    %pause(0.5);

end
toc;
tt = reshape(tt, 1, N+1);