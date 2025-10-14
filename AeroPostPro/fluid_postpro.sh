#!/bin/bash
#===============================================================================
# Post-Processing Script for Aero-F
#===============================================================================
# Step 1: Merge binary outputs into ASCII XPost files
# Step 2: Convert XPost files into a single ExodusII (.exo) file for visualization
#===============================================================================

#-----------------------------#
# User-defined paths & files  #
#-----------------------------#

# Executables
SOWER_BIN="/projects/wang_aoe_lab/KevinWang/Software/sower/bin/sower"
XP2EXO_BIN="/projects/wang_aoe_lab/KevinWang/Software/xp2exo"

# Simulation directory
CASE_DIR="/projects/wang_aoe_lab/ObedIsaac/HardBego_R0mm_14.8kV"
MESH_DIR="$CASE_DIR/Mesh"
RESULTS_DIR="$CASE_DIR/results"

# Mesh and connectivity files
MESH_FILE="$MESH_DIR/OUTPUT.msh"
CON_FILE="$MESH_DIR/OUTPUT.con"

# Output fields
FIELDS=("dpressure" "velocity")

# ExodusII output
ASCII_MESH_FILE="$MESH_DIR/fluid3d.top"
EXO_OUTPUT="fluid3d.exo"

#-----------------------------#
# Step 1: Merge binary outputs #
#-----------------------------#

echo ">>> Step 1: Merging processor outputs into XPost files..."
for FIELD in "${FIELDS[@]}"; do
  echo "    Processing field: $FIELD"
  $SOWER_BIN -fluid -merge \
    -mesh "$MESH_FILE" \
    -con "$CON_FILE" \
    -res "$RESULTS_DIR/$FIELD" \
    -output "$FIELD"
done

#-----------------------------#
# Step 2: Convert to ExodusII  #
#-----------------------------#

# Build list of .xpost files based on FIELDS array
XPOST_FILES=()
for FIELD in "${FIELDS[@]}"; do
  XPOST_FILES+=("${FIELD}.xpost")
done

echo ">>> Step 2: Converting XPost files to ExodusII format..."
$XP2EXO_BIN "$TOPOLOGY_FILE" "$EXO_OUTPUT" "${XPOST_FILES[@]}"

echo ">>> Post-processing complete. Output: $EXO_OUTPUT"
