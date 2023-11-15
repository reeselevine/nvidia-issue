CXX = g++
CXXFLAGS = -std=c++17
CLSPVFLAGS = -cl-std=CL2.0 -inline-entry-points

SHADERS = $(patsubst %.cl,%.spv,$(wildcard *.cl))

.PHONY: clean easyvk 

all: build easyvk runner

build:
	mkdir -p build

clean:
	rm -r build

easyvk: easyvk/src/easyvk.cpp easyvk/src/easyvk.h
	$(CXX) $(CXXFLAGS) -Ieasyvk/src -c easyvk/src/easyvk.cpp -o build/easyvk.o

runner: runner.cpp copy-spv
	$(CXX) $(CXXFLAGS) -Ieasyvk/src build/easyvk.o runner.cpp -lvulkan -o build/runner

copy-spv:
	cp test.spv build

