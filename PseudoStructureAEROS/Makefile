# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /apps/easybuild/software/tinkercliffs-rome/CMake/3.16.4-intel-2019b/bin/cmake

# The command to remove a file.
RM = /apps/easybuild/software/tinkercliffs-rome/CMake/3.16.4-intel-2019b/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kevinw3/tinkercliffs/toys/PseudoStructureAEROS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevinw3/tinkercliffs/toys/PseudoStructureAEROS

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/apps/easybuild/software/tinkercliffs-rome/CMake/3.16.4-intel-2019b/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/apps/easybuild/software/tinkercliffs-rome/CMake/3.16.4-intel-2019b/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/kevinw3/tinkercliffs/toys/PseudoStructureAEROS/CMakeFiles /home/kevinw3/tinkercliffs/toys/PseudoStructureAEROS/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/kevinw3/tinkercliffs/toys/PseudoStructureAEROS/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named cplLib

# Build rule for target.
cplLib: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 cplLib
.PHONY : cplLib

# fast build rule for target.
cplLib/fast:
	$(MAKE) -f CMakeFiles/cplLib.dir/build.make CMakeFiles/cplLib.dir/build
.PHONY : cplLib/fast

#=============================================================================
# Target rules for targets named cplCode

# Build rule for target.
cplCode: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 cplCode
.PHONY : cplCode

# fast build rule for target.
cplCode/fast:
	$(MAKE) -f CMakeFiles/cplCode.dir/build.make CMakeFiles/cplCode.dir/build
.PHONY : cplCode/fast

#=============================================================================
# Target rules for targets named comm

# Build rule for target.
comm: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 comm
.PHONY : comm

# fast build rule for target.
comm/fast:
	$(MAKE) -f Comm/CMakeFiles/comm.dir/build.make Comm/CMakeFiles/comm.dir/build
.PHONY : comm/fast

#=============================================================================
# Target rules for targets named hetero

# Build rule for target.
hetero: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 hetero
.PHONY : hetero

# fast build rule for target.
hetero/fast:
	$(MAKE) -f Hetero/CMakeFiles/hetero.dir/build.make Hetero/CMakeFiles/hetero.dir/build
.PHONY : hetero/fast

Main.o: Main.C.o

.PHONY : Main.o

# target to build an object file
Main.C.o:
	$(MAKE) -f CMakeFiles/cplLib.dir/build.make CMakeFiles/cplLib.dir/Main.C.o
	$(MAKE) -f CMakeFiles/cplCode.dir/build.make CMakeFiles/cplCode.dir/Main.C.o
.PHONY : Main.C.o

Main.i: Main.C.i

.PHONY : Main.i

# target to preprocess a source file
Main.C.i:
	$(MAKE) -f CMakeFiles/cplLib.dir/build.make CMakeFiles/cplLib.dir/Main.C.i
	$(MAKE) -f CMakeFiles/cplCode.dir/build.make CMakeFiles/cplCode.dir/Main.C.i
.PHONY : Main.C.i

Main.s: Main.C.s

.PHONY : Main.s

# target to generate assembly for a file
Main.C.s:
	$(MAKE) -f CMakeFiles/cplLib.dir/build.make CMakeFiles/cplLib.dir/Main.C.s
	$(MAKE) -f CMakeFiles/cplCode.dir/build.make CMakeFiles/cplCode.dir/Main.C.s
.PHONY : Main.C.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... cplLib"
	@echo "... cplCode"
	@echo "... comm"
	@echo "... hetero"
	@echo "... Main.o"
	@echo "... Main.i"
	@echo "... Main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

