function [dudt, S] = FVM_scalar_multiphase_ddt(dt, u, xc, interface_locations, alphas, betas, beta_ts, eq_flag, flux_flag, BC_flag, source_flag)

    n = length(u);
    dx = (xc(end) - xc(1)) / n;

    % Initialize material identification
    if length(alphas) - 1 == length(interface_locations)
        n_mats = length(interface_locations);
        interface_id = zeros(n_mats, n);
        for i = 1:n_mats
            interface_id(i, :) = interface_locations(i) < xc;
        end
        material_id = sum(interface_id, 1) + 1;
    else
        error('Mismatch in number of materials and material parameters');
    end

    % Choose boundary conditions
    if strcmp(BC_flag, 'Zero')
        u = [0, u, 0];
        material_id = [material_id(1), material_id, material_id(end)];
    elseif strcmp(BC_flag, 'Periodic')
        u = [u(end), u, u(1)];
        material_id = [material_id(end), material_id, material_id(1)];
    else
        error('Error: Did not recognize boundary condition flag!');
    end

    % Choose the equation type
    if strcmp(eq_flag, 'Wave')
        fu_bar = [0, zeros(1, n), 0];
        dfdu_bar = [0, zeros(1, n), 0];
        S = [0, zeros(1, n), 0];
        

        for i = 1:n+2
            alpha = alphas(material_id(i));
            beta = betas(material_id(i));
            beta_t = beta_ts(material_id(i));
            
            fu_bar(i) = alpha * u(i);
            dfdu_bar(i) = alpha;
            if source_flag
                S(i) = -beta * max((u(i) - beta_t), 0);
            end
        end
    elseif strcmp(eq_flag, 'Burgers')
        fu_bar = u.^2 / 2;
        dfdu_bar = u;
        S = zeros(1, n);
    else
        error('Error: Did not recognize equation flag!');
    end

    % Flux calculation
    fu_interface = zeros(1, n+1);
    
    if strcmp(flux_flag, 'Central')
        fu_interface(1:end-1) = (fu_bar(1:end-1) + fu_bar(2:end)) / 2;

    elseif strcmp(flux_flag, 'Upwind')
        for i = 1:n  %Needs to be updated (n->n+1 etc.)
            if dfdu_bar(i) >= 0
                fu_interface(i) = fu_bar(i);
            else
                fu_interface(i) = fu_bar(mod(i, n) + 1); % Wrap around for periodic
            end
        end
        fu_interface(end) = fu_bar(1);  % Last interface connects to the first cell

    elseif strcmp(flux_flag, 'Godunov')
        for i = 1:n+1
            left = u(i);
            right = u(i + 1);  % Next cell wraps around
            if left > right
                fu_interface(i) = max(fu_bar(i), fu_bar(i + 1));
            else
                fu_interface(i) = min(fu_bar(i), fu_bar(i + 1));
            end
        end
    else
        error('Error: Did not recognize flux function flag!');
    end

    % Compute du/dt
    S = S(2:end-1);
    dudt = (fu_interface(1:end-1) - fu_interface(2:end)) / dx + S; 
end


