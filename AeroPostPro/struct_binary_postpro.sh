#!/bin/bash
#===============================================================================
# Post-Processing Script for Aero-S (with BINARYOUTPUT = ON)
#===============================================================================
# Step 1: [MUST BE DONE BEFORE RUNNING THIS SCRIPT]
#         Prepare a mesh file containing only "NODES" and "TOPOLOGY" sections,
#         ending with "END" (no SURFACETOPO). Example: mesh.include.postpro
# Step 2: Create binary mesh info (OUTPUT.con, OUTPUT.msh1, etc.)
# Step 3: Convert binary results for each field into ASCII XPost files
# Step 4: Convert XPost files to ExodusII (.exo) for visualization in ParaView
#===============================================================================

#-----------------------------#
# User-defined paths & files  #
#-----------------------------#

# Executables
SOWER_BIN="/projects/wang_aoe_lab/KevinWang/Software/sower/bin/sower"
AEROS_BIN="/projects/wang_aoe_lab/KevinWang/Codes/FEMWorking/bin/aeros"
XP2EXO_BIN="/projects/wang_aoe_lab/KevinWang/Software/xp2exo"

# Simulation directory
CASE_DIR="/projects/wang_aoe_lab/ObedIsaac/HardBego_R0mm_14.8kV"
RESULTS_DIR="$CASE_DIR/results"

# Mesh and decomposition files
MESH_INCLUDE="mesh.include.postpro"
DEC_FILE="$CASE_DIR/solid3d.optDec"

# Number of processors for decomposition
NCPU=16

# Output mesh & connectivity files (produced in Step 2)
CON_FILE="OUTPUT.con"
MSH_FILE="OUTPUT.msh"

# Solution fields (binary result files in RESULTS_DIR)
FIELDS=("disp.dat" "stressp1.dat")

# ExodusII output
TOPOLOGY_FILE="${MESH_INCLUDE}.top"
EXO_OUTPUT="solid3d.exo"

#-----------------------------#
# Step 2: Create binary mesh info
#-----------------------------#

echo ">>> Step 2: Creating binary mesh info..."
$SOWER_BIN -struct -mesh "$MESH_INCLUDE" -dec "$DEC_FILE" -cpu "$NCPU" -cluster 1

#-----------------------------#
# Step 3: Convert binary results to XPost
#-----------------------------#

echo ">>> Step 3: Converting binary results to XPost format..."
for FIELD in "${FIELDS[@]}"; do
  FIELD_PATH="$RESULTS_DIR/$FIELD"
  echo "    Processing field: $FIELD_PATH"
  $SOWER_BIN -struct -merge \
    -con "$CON_FILE" \
    -mesh "$MSH_FILE" \
    -result "$FIELD_PATH"
done

echo ">>> Step 3 complete. XPost files written to $RESULTS_DIR."

#-----------------------------#
# Step 4: Convert XPost to ExodusII
#-----------------------------#

# Step 4.1: Generate topology file
echo ">>> Step 4.1: Generating .top file..."
$AEROS_BIN -t "$MESH_INCLUDE"

# Step 4.2: Convert XPost files to ExodusII
echo ">>> Step 4.2: Converting XPost files to ExodusII format..."
XPOST_FILES=()
for FIELD in "${FIELDS[@]}"; do
  XPOST_FILES+=("$RESULTS_DIR/${FIELD}.xpost")
done

$XP2EXO_BIN "$TOPOLOGY_FILE" "$EXO_OUTPUT" "${XPOST_FILES[@]}"

echo ">>> Post-processing complete. Output file: $EXO_OUTPUT"


