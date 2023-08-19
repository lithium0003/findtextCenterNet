all:
	$(CPP) /std:c++20 /O2 /Fe:linedetect.exe main.cpp
	copy linedetect.exe ..

clean:
	rm -f linedetect.exe
