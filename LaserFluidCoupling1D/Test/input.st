// Test simulation
under LaserFluid1D {
  Xmin = 0.0;
  Xmax = 1.0;
  NumberOfNodes = 401;
  SourceIntensity = 10.0;
  Alpha0 = 1.0;

  Interval1_Xmin = 0.25;
  Interval1_Xmax = 0.5;
  Interval1_Alpha = 4.0;

  Interval2_Xmin = 0.5;
  Interval2_Xmax = 0.75;
  Interval2_Alpha = 0.25;

  ResultFolder = "./";
  ResultFilePrefix = "output";
}
