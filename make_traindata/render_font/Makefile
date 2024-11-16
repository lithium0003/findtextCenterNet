all: render_font test_font

render_font: render_font.cpp 
	g++ -O2 -march=native render_font.cpp -o render_font -std=c++17 `pkg-config --cflags --libs freetype2` 

test_font: test_font.cpp 
	g++ -O2 -march=native test_font.cpp -o test_font -std=c++17 `pkg-config --cflags --libs freetype2` 

.PHONY : clean
clean: 
	@rm -rf render_font test_font
