under Input {

  Processors = 256;

  under Solver[0] { //Aero-H
    ScalingLaw = Amdahl;
    ScalingLawCoefficient = 0.62;
    PhysicalTimeStep = 50.0;
    WallTimePerStepOneCore = 38.9;
  }

  under Solver[1] { //Aero-F
    ScalingLaw = Amdahl;
    ScalingLawCoefficient = 0.99983;
    PhysicalTimeStep = 1.0;
    WallTimePerStepOneCore = 188.0;
  }

  under Solver[2] { //Aero-S
    ScalingLaw = Amdahl;
    ScalingLawCoefficient = 0.82;
    PhysicalTimeStep = 100.0;
    WallTimePerStepOneCore = 410.0;
  }
}
