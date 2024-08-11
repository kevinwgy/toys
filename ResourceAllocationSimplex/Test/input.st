under Input {

  Processors = 50000;

  under Solver[0] {
    ScalingLaw = Amdahl;
    ScalingLawCoefficient = 0.999;
    PhysicalTimeStep = 0.2;
    WallTimePerStepOneCore = 0.2;
  }

  under Solver[1] {
    ScalingLaw = Gustafson;
    ScalingLawCoefficient = 0.1;
    PhysicalTimeStep = 1.2;
    WallTimePerStepOneCore = 0.5;
  }

  under Solver[2] {
    ScalingLaw = Amdahl;
    ScalingLawCoefficient = 0.9995;
    PhysicalTimeStep = 0.2;
    WallTimePerStepOneCore = 1.0;
  }

  under Solver[3] {
    ScalingLaw = Perfect;
    PhysicalTimeStep = 0.1;
    WallTimePerStepOneCore = 0.5;
  }

  under Solver[4] {
    ScalingLaw = Gustafson;
    ScalingLawCoefficient = 0.2;
    PhysicalTimeStep = 0.3;
    WallTimePerStepOneCore = 0.8;
  }

  under Solver[5] {
    ScalingLaw = Gustafson;
    ScalingLawCoefficient = 0.1;
    PhysicalTimeStep = 0.4;
    WallTimePerStepOneCore = 1.2;
  }

  under Solver[6] {
    ScalingLaw = Gustafson;
    ScalingLawCoefficient = 0.05;
    PhysicalTimeStep = 0.06;
    WallTimePerStepOneCore = 4.5;
  }
}
