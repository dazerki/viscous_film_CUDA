# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/888/bin/cmake

# The command to remove a file.
RM = /snap/cmake/888/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/antoine/Documents/Viscous/GPU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/antoine/Documents/Viscous/GPU/build

# Include any dependencies generated for this target.
include CMakeFiles/project.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/project.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/project.dir/flags.make

CMakeFiles/project.dir/src/viscous.c.o: CMakeFiles/project.dir/flags.make
CMakeFiles/project.dir/src/viscous.c.o: ../src/viscous.c
CMakeFiles/project.dir/src/viscous.c.o: CMakeFiles/project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/antoine/Documents/Viscous/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/project.dir/src/viscous.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/project.dir/src/viscous.c.o -MF CMakeFiles/project.dir/src/viscous.c.o.d -o CMakeFiles/project.dir/src/viscous.c.o -c /home/antoine/Documents/Viscous/GPU/src/viscous.c

CMakeFiles/project.dir/src/viscous.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/project.dir/src/viscous.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/antoine/Documents/Viscous/GPU/src/viscous.c > CMakeFiles/project.dir/src/viscous.c.i

CMakeFiles/project.dir/src/viscous.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/project.dir/src/viscous.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/antoine/Documents/Viscous/GPU/src/viscous.c -o CMakeFiles/project.dir/src/viscous.c.s

CMakeFiles/project.dir/src/gl_utils.cpp.o: CMakeFiles/project.dir/flags.make
CMakeFiles/project.dir/src/gl_utils.cpp.o: ../src/gl_utils.cpp
CMakeFiles/project.dir/src/gl_utils.cpp.o: CMakeFiles/project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/antoine/Documents/Viscous/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/project.dir/src/gl_utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/project.dir/src/gl_utils.cpp.o -MF CMakeFiles/project.dir/src/gl_utils.cpp.o.d -o CMakeFiles/project.dir/src/gl_utils.cpp.o -c /home/antoine/Documents/Viscous/GPU/src/gl_utils.cpp

CMakeFiles/project.dir/src/gl_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/project.dir/src/gl_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/antoine/Documents/Viscous/GPU/src/gl_utils.cpp > CMakeFiles/project.dir/src/gl_utils.cpp.i

CMakeFiles/project.dir/src/gl_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/project.dir/src/gl_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/antoine/Documents/Viscous/GPU/src/gl_utils.cpp -o CMakeFiles/project.dir/src/gl_utils.cpp.s

CMakeFiles/project.dir/src/kernel.cu.o: CMakeFiles/project.dir/flags.make
CMakeFiles/project.dir/src/kernel.cu.o: ../src/kernel.cu
CMakeFiles/project.dir/src/kernel.cu.o: CMakeFiles/project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/antoine/Documents/Viscous/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/project.dir/src/kernel.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/project.dir/src/kernel.cu.o -MF CMakeFiles/project.dir/src/kernel.cu.o.d -x cu -c /home/antoine/Documents/Viscous/GPU/src/kernel.cu -o CMakeFiles/project.dir/src/kernel.cu.o

CMakeFiles/project.dir/src/kernel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/project.dir/src/kernel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/project.dir/src/kernel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/project.dir/src/kernel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/project.dir/src/project.cu.o: CMakeFiles/project.dir/flags.make
CMakeFiles/project.dir/src/project.cu.o: ../src/project.cu
CMakeFiles/project.dir/src/project.cu.o: CMakeFiles/project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/antoine/Documents/Viscous/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/project.dir/src/project.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/project.dir/src/project.cu.o -MF CMakeFiles/project.dir/src/project.cu.o.d -x cu -c /home/antoine/Documents/Viscous/GPU/src/project.cu -o CMakeFiles/project.dir/src/project.cu.o

CMakeFiles/project.dir/src/project.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/project.dir/src/project.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/project.dir/src/project.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/project.dir/src/project.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target project
project_OBJECTS = \
"CMakeFiles/project.dir/src/viscous.c.o" \
"CMakeFiles/project.dir/src/gl_utils.cpp.o" \
"CMakeFiles/project.dir/src/kernel.cu.o" \
"CMakeFiles/project.dir/src/project.cu.o"

# External object files for target project
project_EXTERNAL_OBJECTS =

project: CMakeFiles/project.dir/src/viscous.c.o
project: CMakeFiles/project.dir/src/gl_utils.cpp.o
project: CMakeFiles/project.dir/src/kernel.cu.o
project: CMakeFiles/project.dir/src/project.cu.o
project: CMakeFiles/project.dir/build.make
project: /usr/lib/x86_64-linux-gnu/libpython3.8.so
project: /usr/lib/x86_64-linux-gnu/libGLEW.so
project: /usr/lib/x86_64-linux-gnu/libglfw.so
project: /usr/lib/x86_64-linux-gnu/libOpenGL.so
project: /usr/lib/x86_64-linux-gnu/libGLX.so
project: /usr/lib/x86_64-linux-gnu/libGLU.so
project: CMakeFiles/project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/antoine/Documents/Viscous/GPU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/project.dir/build: project
.PHONY : CMakeFiles/project.dir/build

CMakeFiles/project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/project.dir/clean

CMakeFiles/project.dir/depend:
	cd /home/antoine/Documents/Viscous/GPU/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/antoine/Documents/Viscous/GPU /home/antoine/Documents/Viscous/GPU /home/antoine/Documents/Viscous/GPU/build /home/antoine/Documents/Viscous/GPU/build /home/antoine/Documents/Viscous/GPU/build/CMakeFiles/project.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/project.dir/depend

