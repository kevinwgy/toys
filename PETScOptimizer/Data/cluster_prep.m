
clear
clc
close all

%% original

%% LasInterm
pce_path = '/Users/shaowenluo/Dropbox/Research/Daniel_Shaowen_shared/Network Project/Data/PCE/';
data_path = '/Users/shaowenluo/Dropbox/Research/Daniel_Shaowen_shared/Network Project/Data/71industry/';
%% Load shocks and data
N=66; % number of sectors
% vector of changes in total hours by industry
%temp_path = fullfile(data_path,'BLS_labor_shock_202108.xls');

%BLS_shock_raw = readmatrix(temp_path);
%BLS_shock = BLS_shock_raw(1:N,4); % diff_2005

% vector of changes PCE spending

temp_path = fullfile(pce_path,'PCE_shock_202206.xls');

PCE_shock_raw = readmatrix(temp_path);
PCE_shock = PCE_shock_raw(1:N,4); % diff_2005_pce

% vector of hourly wage changes by industry

temp_path = fullfile(data_path,'wage_ppi_use.xlsx');

price_raw = readmatrix(temp_path);
wage_data = price_raw(1:N,6); % w_annual change 2006
% Copy numbers in row 49 (HS) and paste them to row 50 (ORE)
% Both HS (Housing) and ORE (Other real estate) in BEA code are classified as
% 531 (Real estate) in NAICS code
wage_data(49) = wage_data(48);
PPI_data = price_raw(1:N,4); % p_annual change 2006
PPI_data(49) = PPI_data(48);

%% Import IO data matrix

%% Calibrate the IO Table and build Omega matrix
% year for which to run the calibration.
FileName   = 'io71.mat';
File       = fullfile(data_path, FileName);
load(File)
year = 2019;

eval(['supply_raw = supply_' num2str(year) ';'])
eval(['use_raw = use_' num2str(year) ';'])

makeuse = supply_raw(1:N,1:N);
useuse = use_raw(1:N,1:N);
empuse = use_raw(75,1:N)'; % V001 Compensation of employees
gos = use_raw(76,1:N)'; % V003 gross operating surplus

Pce = use_raw(1:N,73); % F06C Personal consumption expenditures
Pce(isnan(Pce)) = 0;

va_sales = empuse + gos;
int_sales = sum(useuse)';

int_share=int_sales./(int_sales+va_sales);
va_share=va_sales./(int_sales+va_sales);

alphaL=empuse./va_sales;
alphaK=gos./va_sales;

%% construct IO
% use and make tables are by industry-commodity, convert them to industry-industry
% follow Pasten JME
%N = size(makeuse,1);
II = ones(N,N);
% the market share (“SHARE”) of industry i s production of commodity c as
Share = makeuse.*((II*makeuse).^(-1));
% multiply the share and use tables to calculate the dollar amount
% that industry i sells to industry j
Revshare = Share*useuse;
% the revenue share matrix to calculate the percentage of industry j’s inputs purchased
% from industry i and label the resulting matrix SUPPS HARE:
% I think there is a typo in Pasten's paper, II*Use instead of Use*II, and
% no transpose
Suppshare = Revshare.*((II*useuse).^(-1)); % j use of i
Omega = Suppshare'; % i use of j

%% finalshare
finaltemp = Share*Pce;
beta = finaltemp./(sum(Pce));

%Tot_int=sum(IO)';
%Tot_use = supply_raw(74,1:N)'; % total industry supply
%Final=(Tot_use-Tot_int);
%beta = Final(1:N);
%beta(isnan(beta)) = 0;
%beta(beta<0) = 0; %remove industries with negative implied final sales
%beta = beta/sum(beta);

%beta = Pce/sum(Pce);

average_mu = 1;
mu = 1;

%% Write Omega in standard form - relabeling
%Omega_re= final consumers + N final produc.+ N va. produc. + N int. produc. +  2FACTORS

F=2*N;
D=1+3*N+F+3; 
% 1 is consumption today, 2:1+N is goods today, N+2:2*N+1 is VA today,
% 2*N+2:3*N+1 is intermediates today, 3*N+2:4*N+1 is labor today, 
%4*N+2:5*N+1 is capital today, 5*N+2 is HtM consumer, 5*N+3 is the Ricardian
%consumer, 5*N+4 is (aggregate) consumption good tomorrow. 

Omega_re= zeros(D,D);
Omega_re(1,2:N+1)= beta;
Omega_re(2:1+N,1+N+1:2*N+1) = diag(mu.^(-1))*diag(va_share);
Omega_re(2:1+N,1+2*N+1:3*N+1) = diag(mu.^(-1))*diag(int_share);
for x = 1:N
Omega_re(1+N+x, 1+3*N+x)=alphaL(x)/(alphaL(x)+alphaK(x));
Omega_re(1+N+x, 1+4*N+x)=alphaK(x)/(alphaL(x)+alphaK(x));
end
Omega_re(1+2*N+1:1+3*N,2:N+1) = Omega;

Omega_re(1+5*N+1,end)=1;
Omega_re(1+5*N+2,1)=0.5;
Omega_re(1+5*N+2,end)=.5; 


%% Create Omega_tilde

one=ones(1,1);
mu_ones= ones(D,1);
in_mu=mu_ones;
Omega_tilde= diag(in_mu)*Omega_re;

Psi_re = inv(eye(D)-Omega_re);
Psi_tilde = inv(eye(D)-Omega_tilde);

%Create a categorical variable to set if conditions in constraints in AMPL
factor=ones(D,1); %one for goods
factor(1+3*N+1:1+5*N, 1) = 0; % zero for factors
factor(end) = 0; % Since consumption tomorrow is treated like a factor. 
factor(end-2) = 3; % since the second and third to last rows are consumers. 
factor(end-1) = 2; % since the second and third to last rows are consumers. 

factor2=zeros(D,1); %one for goods
factor2(2:N+1)=1; %one for goods

keynes = ones(D,1);  % zero means flexible, -1 means sticky. 
keynes(1+4*N+1:1+5*N, 1) = 0; %zero for flexible factors: capital
keynes(1+5*N+3, 1) = 0; % zero for tomorrow's consumption
%keynes(1+3*N+1:1+4*N, 1) =-1; %-1 negative one for sticky factors: labor.
keynes(1+3*N+1:1+4*N, 1) = 0; %if labor is not sticky

element = zeros(D,1);
element(1) = 0; % agg consumption
element(2:67) = 1; % goods today
element(68:133) = 2; % VA today
element(134:199) = 3; % intermediates today
element(200:265) = 4; % labor today
element(266:331) = 5; % capital today
element(332) = 6; % HtM consumer
element(333) = 7; % Ricardian consumer
element(334) = 8; % agg consumption tmr

%temp_factor = factor;
%clearvars -except factor lambdaL lambdaK in_mu mu Ind F D N A L Omega_re theta IO Sigma va_share int_share alphaL alphaK


%assign ownership of the factors
chi = (factor==0).*rand(D,1);
chi(end) = 0; % tomorrow's consumption is spent by Ricardian household. 
chi(1+4*N+1:1+5*N, 1)=0; % HtM don't own capital
chi(1+3*N+1:1+4*N, 1)=0; % HtM don't own capital
 
% Determin price elasticity of factor supply
phi = zeros(D,1); 
phi(1+3*N+1:1+4*N, 1) = 0; % labor 
phi(1+4*N+1:1+5*N, 1) = 0; % capital 

cobb_douglas = 2*ones(D,1); 
cobb_douglas(factor == 1) = 0;
cobb_douglas(1) = 1;
cobb_douglas(5*N+3) = 1; 

theta1 = .2*ones(N,1); % across int
epsilon = .6*ones(N,1); % VA v inter
sigma = 1;
eta = .6*ones(N,1);  % across VA


% frequency
FileName   = 'freq.mat';
File       = fullfile(data_path, FileName);
load(File)
freq_re = ones(D,1);
freq_re(2:N+1) = freq;

%Choose elasticity parameters - Create vector of elast.

rho = 1;
factor_ela=.2*ones(F,1);
theta = cat(1,sigma,epsilon,eta, theta1,factor_ela,.95,rho,.95);

s = 1; %htm_loop, =1, no htm

htm_share = 0.2*(s-1);

% Initiate the model

Domar = Psi_re(1,2:N+1);
h = 10^(-2);

init_lambda = Psi_re(1,:)';
init_lambda(1) = 1;
init_lambda(end-2) = (Psi_re(1,:)*chi)*init_lambda(1);
init_lambda(end-1) = 2*(1-(Psi_re(1,:)*chi))*init_lambda(1);
init_lambda(end) = 0.5*init_lambda(end-1);
init_lambda1 = init_lambda;
init_p = ones(D,1);

phi_htm = ones(D,1);
phi_htm(1+3*N+1:1+4*N, 1) =1-htm_share;

% Choose A and mu

A = ones(D,1);
B = ones(D,D);

%shock = -BLS_shock;

t = 1;
counter = 1;

%A(1+3*N+1:1+4*N) =(1-t*shock); % sectoral supply shock
B(1,2:N+1)=(1-t*.66)+t*.66*(1+PCE_shock); % sectoral demand shock
B(5*N+3,end)=1+min(counter-1,1)*.105; % aggregate demand shock

B(1,:) = B(1,:)/sum(Omega_re(1,:)*B(1,:)');
B(5*N+3,:)=B(5*N+3,:)/sum(Omega_re(5*N+3,:)*B(5*N+3,:)');

in_mu=ones(D,1);
mu=ones(D,1);


% price target
tar_p = ones(D,1);
tar_p(2:67) = PPI_data + 1;
tar_p(200:265) = wage_data + 1;


%% save data
writematrix(tar_p, 'tar_p.txt')
writematrix(A, 'A.txt')
writematrix(B, 'B.txt')
writematrix(freq_re, 'freq_re.txt')
writematrix(phi_htm, 'phi_htm.txt')
writematrix(theta, 'theta.txt')
writematrix(Omega_re, 'Omega_re.txt')
writematrix(keynes, 'keynes.txt')
writematrix(element, 'element.txt')
writematrix(cobb_douglas, 'cobb_douglas.txt')
writematrix(factor, 'factor.txt')
writematrix(factor2, 'factor2.txt')
writematrix(init_p, 'init_p.txt')
writematrix(init_lambda, 'init_lambda.txt')


