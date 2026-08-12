# update pkg-config
ifneq ($(PKGS),)
CPPFLAGS += $(shell pkg-config --cflags $(PKGS))
LIBS  += $(shell pkg-config --libs $(PKGS))
endif

# common
INCS := $(wildcard *.h) $(wildcard *.hpp) $(wildcard *.cuh)
SRCS := $(wildcard *.c) $(wildcard *.cpp) $(wildcard *.cu)
OBJS := $(SRCS:.c=.o)
OBJS := $(OBJS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CPPFLAGS) $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CPPFLAGS) $(CXXFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ $(CPPFLAGS) $(CUFLAGS) $<
