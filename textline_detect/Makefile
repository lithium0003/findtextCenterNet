TARGET    = linedetect
SRCDIR    = ./src

CXXFLAGS += -std=c++20 -O3 -MMD -MP
LDFLAGS  +=
LIBS     +=

SOURCES   = $(wildcard $(SRCDIR)/minpack/*.cpp)
SOURCES  += $(wildcard $(SRCDIR)/*.cpp)
OBJECTS   = $(SOURCES:.cpp=.o)
DEPENDS   = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	-rm -f $(OBJECTS) $(DEPENDS)

distclean:
	-rm -f $(TARGET)

-include $(DEPENDS)