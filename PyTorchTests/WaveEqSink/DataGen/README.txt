Running the simulations:
________________________
	- Ensure the proper module(s) is loaded:
		module load $LMOD_SYSTEM_NAME/matlab/R2021a
	- To execute the simulation
		matlab -nodisplay -nosplash -nodesktop -nosoftwareopengl  -batch  main_sim -logfile log.out


Explaining the results.txt file:
________________________________

	- Each row of the file records the solutions of each simulation.
	- The first value is the 'simcode' which is an unique identifier for each simulation
	- In these sets of simulations 3 parameters have been varied:
		* a -> initial area under the input wave representing the strength of the wave
		* \beta(2) -> scaling coefficient for sink term of second material
		* \beta_t(2) -> threshold coefficient for sink term of second material
	- The simcode is arranged as follows: (a X 10) _ (\beta(2) X 10^3) _ (\beta_t(2) X 10^3)
	- The results file is arranged as follows with 'n' number of 'materials':
		* column 1 -> simcode
		* column 2 -> computation time of the simulation
		* column 3 -> a
		* column 4 -> \beta(2)
		* column 5 -> \beta_t(2)
		* column 6:(5+n) -> \int{U(t = t_f)} for each material
		* column (6+n):(5+2n) -> \int{S(t = t_f)} for each material
