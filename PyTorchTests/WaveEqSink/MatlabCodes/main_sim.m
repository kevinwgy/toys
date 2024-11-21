clear; clc; close all;
tstart = tic;
poolobj = parpool("local", 64);

%%%%%%%%%%%%%%% Setting up simulation variables
N_vars = [10 10 10]; numsims = prod(N_vars);
a_vec = linspace(1, 2, N_vars(1));
beta2_vec = linspace(0.005, 0.015, N_vars(2));
betat2_vec = linspace(0.15, 0.25, N_vars(3));
% adding random noise to the input space
%a_vec = a_vec + mean(diff(a_vec))*(rand(size(a_vec)) - 0.5)*0.5;
%beta2_vec = beta2_vec + mean(diff(beta2_vec))*(rand(size(beta2_vec)) - 0.5)*0.5;
%betat2_vec = betat2_vec + mean(diff(betat2_vec))*(rand(size(betat2_vec)) - 0.5)*0.5;

% Generating a Cartesian product of the variable inputs and shuffling the rows
X = cartprod(a_vec, beta2_vec, betat2_vec);
X = X(randperm(numsims), :);

N = 2^12;                            % #of gridpoints
tf = 200;                            % final time

%%%%%%%%%%%%%%% Clear results %%%%%%%%%%%%%%%%%%%
if exist('results.txt', 'file')
    delete results.txt
end

parfor sim_number = 1:numsims
          tsim = tic;
          a = X(sim_number, 1);beta2= X(sim_number, 2); betat2= X(sim_number, 3);
          simcode = [num2str(round(a*10)) '_' num2str(round(beta2*10^3)) '_' num2str(round(betat2*10^3))];
          disp(['launching code: ' simcode])
          %%%% Set-up domain and ICs
          x = linspace(0, 100, N);             % location of gridpoints
          xc = 0.5 * (x(1:end-1) + x(2:end));  % location of cell centers
          u0 = a*normpdf(xc,3,1);                % initial condition (Gaussian wave)
          %%%% initialize parameters
          S = zeros(size(xc));
          dx = (x(end) - x(1))/N; CFL_max = 0.1;
          dt = CFL_max * dx * max(abs(u0)); t = 0:dt:tf;
          u = u0;
        
          %%%% initialize data saving variables
        nsteps = ceil(tf/dt);
        save_dt = 0.1; save_i = 1;
        t_specified = 0:save_dt:t(end);
        t_closest = zeros(size(t_specified));
        for i = 2:length(t_specified)
            [~, ind] = min(abs(t_specified(i) - t));
            t_closest(i) = t(ind);
        end
          U_mat = zeros(length(t_specified), length(xc));
          S_mat = zeros(length(t_specified), length(xc));
          E_vec = zeros(length(t_specified), 1);
          t_vec = zeros(length(t_specified), 1);
        
          %%%% multiphase parameters
          interface_locations = [10 50];
          alphas = [1 2 0.5];
          betas = [0.025 beta2 0.2]; 
          beta_ts= [0.3 betat2 0.2];
          %%%% Select computation schemes with flags
          eq_flag = 'Wave'; flux_flag = 'Godunov'; BC_flag = 'Periodic';
        
          for i = 1:length(t)
          %%%%%%%%%%%%%%%%%%% time stepping loop %%%%%%%%%%%%%%%%%%%%%%%%%%
        
              [dudt, s] = FVM_scalar_multiphase_ddt(dt, u, xc, interface_locations, alphas, betas, beta_ts, eq_flag, flux_flag, BC_flag, 1);      
              u = u + dt*dudt;              % forward Euler propagation of u
              S = S + s*dt;                 % source/sink
              E = dx*(sum(u) - sum(S));     % conserved quantity
          %%%%%%%%%%%%%%%%%%%%%%%% saving data locally %%%%%%%%%%%%%%%%%%%%%%%%
              if any(t_closest == t(i))
                  U_mat(save_i, :) = u;
                  S_mat(save_i, :) = S;
                  E_vec(save_i) = E;
                  t_vec(save_i) = t(i);
                  save_i = save_i+1;
              end
          end
        
          %%%%%%%%%%%%%%%%%%% integrating energies %%%%%%%%%%%%%%%%%%%%%%%%%%%
          [int_U, int_S] = energy_integrator(U_mat, S_mat, xc, interface_locations, simcode);
        
          ctime = toc(tsim);
          disp(['code #:' simcode ' terminated after ' num2str(ctime) 's of computation'])
        
          %%%%%%%%%%%%%%%%%%% saving data %%%%%%%%%%%%%%%%%%%%%%%
          output = [];
          output.U = U_mat;
          output.S = S_mat;
          output.E = E_vec;
          output.x = xc;
          output.int_loc = interface_locations;
          output.t = t_vec;
          output.ctime = ctime;
          output.int_U = int_U;
          output.int_S = int_S;
          parsave(simcode, output);
         
          var = [ctime, a, beta2, betat2, int_U(end, :), int_S(end, :), sum(int_U(end, :)) - sum(int_S(end, :))];
          textfile_writer(simcode, var); 
          disp(['code #:' simcode ' terminated after ' num2str(ctime) 's of computation'])
end
delete(poolobj)
CTIME = toc(tstart);
disp(['All ' num2str(numsims)  ' simulations have been completed. Total computation time: ' num2str(CTIME) 's.'])
quit
