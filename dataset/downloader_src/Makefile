all: downloader

downloader: main.cpp 
	g++ -O3 -march=native main.cpp -o downloader -std=c++17 `pkg-config --cflags --libs libcurl` 

.PHONY : clean
clean: 
	@rm -rf downloader
