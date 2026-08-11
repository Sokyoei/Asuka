# update pkg-config
ifneq ($(PKGS),)
CPPFLAGS += $(shell pkg-config --cflags $(PKGS))
LIBS  += $(shell pkg-config --libs $(PKGS))
endif

# common
%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CPPFLAGS) $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CPPFLAGS) $(CXXFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ $(CPPFLAGS) $(CUFLAGS) $<
