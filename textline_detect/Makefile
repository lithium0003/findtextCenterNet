TARGET    = linedetect
SRCDIR    = ./src
OBJDIR    = ./obj

CFLAGS   += -std=c++20 -O3 -march=native -MMD -MP
LDFLAGS  +=
LIBS      =

SOURCES   = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS   = $(addprefix $(OBJDIR)/, $(notdir $(SOURCES:.cpp=.o)))
DEPENDS   = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-mkdir -p $(OBJDIR)
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ -c $<

clean:
	-rm -f $(OBJECTS) $(DEPENDS) $(TARGET)

-include $(DEPENDS)