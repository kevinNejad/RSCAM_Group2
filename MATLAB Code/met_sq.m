function  sol = met_sq
% MET_SQ.M    Plot mean exit time for 
%             mean-reverting sqrt process.
%             Use bvp4c to solve ODE numerically.
%
%             Endpoints are a = 1, b= 2.
%
% creates pic_metsqbig.eps
% when called after met_sqbig.m
%
% DJH Aug 2005
%


a = 1;
b = 2;
lambda = 1;
mu = 0.5;
sigma = 0.3;

solinit = bvpinit(linspace(a,b,100),@sqinit);
options = bvpset('RelTol',1e-8,'AbsTol',1e-8);
sol = bvp4c(@sq,@sqbc,solinit,options);
plot(sol.x,sol.y(1,:),'r--','LineWidth',4)
%plot(sol.x,sol.y(1,:),'g--','LineWidth',4)
%xlabel('Initial data, x','FontSize',20,'FontWeight','Bold')
%ylabel('Mean exit time','FontSize',20,'FontWeight','Bold')
%grid on
set(gca,'FontSize',12)
set(gca,'FontWeight','Bold')
xlabel('x','FontSize',16)
ylabel('u(x)','FontSize',16)


     function yprime = sq(x,y)
     %
     ssq = sigma^2;
     f = lambda*(mu-x); 
     g = sigma*sqrt(x);
     hgs = 0.5*g^2;
     yprime = [y(2); (-1-f*y(2))/hgs];
     end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = sqbc(ya,yb)
%
   res = [ya(1); yb(1)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yinit = sqinit(x)
%
   yinit = [sin(pi*(x-1)); pi*cos(pi*(x-1))];
end
