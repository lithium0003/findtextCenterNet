TARGET    = linedetect.exe

CFLAGS   = /std:c++20 /O2 /utf-8

$(TARGET): src/*.cpp
	$(CXX) -o $@ $** $(CFLAGS)

clean:
	del /F *.obj $(TARGET)