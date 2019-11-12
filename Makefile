CXX ?= g++

CXXFLAGS += -c -Wall -std=c++11 $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs opencv)

all: lane_detection_FPT

lane_detection_FPT: main.o; $(CXX) $< -o $@ $(LDFLAGS)

%.o: %.cpp %.h; $(CXX) $< -o $@ $(CXXFLAGS)

clean: ; rm -f main.o lane_detection_FPT
