# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/e/cpp_MINST

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/e/cpp_MINST/build

# Include any dependencies generated for this target.
include CMakeFiles/NeuralNetworkProject.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/NeuralNetworkProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NeuralNetworkProject.dir/flags.make

CMakeFiles/NeuralNetworkProject.dir/main.cpp.o: CMakeFiles/NeuralNetworkProject.dir/flags.make
CMakeFiles/NeuralNetworkProject.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/cpp_MINST/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/NeuralNetworkProject.dir/main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNetworkProject.dir/main.cpp.o -c /mnt/e/cpp_MINST/main.cpp

CMakeFiles/NeuralNetworkProject.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetworkProject.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/cpp_MINST/main.cpp > CMakeFiles/NeuralNetworkProject.dir/main.cpp.i

CMakeFiles/NeuralNetworkProject.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetworkProject.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/cpp_MINST/main.cpp -o CMakeFiles/NeuralNetworkProject.dir/main.cpp.s

CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.o: CMakeFiles/NeuralNetworkProject.dir/flags.make
CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.o: ../dataloader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/cpp_MINST/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.o -c /mnt/e/cpp_MINST/dataloader.cpp

CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/cpp_MINST/dataloader.cpp > CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.i

CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/cpp_MINST/dataloader.cpp -o CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.s

CMakeFiles/NeuralNetworkProject.dir/layer.cpp.o: CMakeFiles/NeuralNetworkProject.dir/flags.make
CMakeFiles/NeuralNetworkProject.dir/layer.cpp.o: ../layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/cpp_MINST/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/NeuralNetworkProject.dir/layer.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNetworkProject.dir/layer.cpp.o -c /mnt/e/cpp_MINST/layer.cpp

CMakeFiles/NeuralNetworkProject.dir/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetworkProject.dir/layer.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/cpp_MINST/layer.cpp > CMakeFiles/NeuralNetworkProject.dir/layer.cpp.i

CMakeFiles/NeuralNetworkProject.dir/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetworkProject.dir/layer.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/cpp_MINST/layer.cpp -o CMakeFiles/NeuralNetworkProject.dir/layer.cpp.s

CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.o: CMakeFiles/NeuralNetworkProject.dir/flags.make
CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.o: ../neuralnetwork.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/cpp_MINST/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.o -c /mnt/e/cpp_MINST/neuralnetwork.cpp

CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/e/cpp_MINST/neuralnetwork.cpp > CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.i

CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/e/cpp_MINST/neuralnetwork.cpp -o CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.s

# Object files for target NeuralNetworkProject
NeuralNetworkProject_OBJECTS = \
"CMakeFiles/NeuralNetworkProject.dir/main.cpp.o" \
"CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.o" \
"CMakeFiles/NeuralNetworkProject.dir/layer.cpp.o" \
"CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.o"

# External object files for target NeuralNetworkProject
NeuralNetworkProject_EXTERNAL_OBJECTS =

NeuralNetworkProject: CMakeFiles/NeuralNetworkProject.dir/main.cpp.o
NeuralNetworkProject: CMakeFiles/NeuralNetworkProject.dir/dataloader.cpp.o
NeuralNetworkProject: CMakeFiles/NeuralNetworkProject.dir/layer.cpp.o
NeuralNetworkProject: CMakeFiles/NeuralNetworkProject.dir/neuralnetwork.cpp.o
NeuralNetworkProject: CMakeFiles/NeuralNetworkProject.dir/build.make
NeuralNetworkProject: CMakeFiles/NeuralNetworkProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/e/cpp_MINST/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable NeuralNetworkProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NeuralNetworkProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NeuralNetworkProject.dir/build: NeuralNetworkProject

.PHONY : CMakeFiles/NeuralNetworkProject.dir/build

CMakeFiles/NeuralNetworkProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NeuralNetworkProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NeuralNetworkProject.dir/clean

CMakeFiles/NeuralNetworkProject.dir/depend:
	cd /mnt/e/cpp_MINST/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/e/cpp_MINST /mnt/e/cpp_MINST /mnt/e/cpp_MINST/build /mnt/e/cpp_MINST/build /mnt/e/cpp_MINST/build/CMakeFiles/NeuralNetworkProject.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NeuralNetworkProject.dir/depend

