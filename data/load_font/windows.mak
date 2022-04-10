all: load_font.obj
	link.exe c:\freetype2\lib\freetype.lib load_font.obj

load_font.obj:
	cl.exe /c load_font.cpp /I c:\freetype2\include
