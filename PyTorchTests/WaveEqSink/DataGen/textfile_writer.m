function textfile_writer(simcode, var)
str = simcode;
nvar = length(var);
for i = 1:nvar
  str = [str, sprintf('     %12.8f', var(i))];
end
str = [str, newline];
  fileID = fopen('results.txt', 'a');
  fprintf (fileID, str);
  fclose(fileID);

end