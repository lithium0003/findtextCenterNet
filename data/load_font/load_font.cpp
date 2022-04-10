#include <stdio.h>
#include <string.h>
#include <sstream>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TRUETYPE_TABLES_H
#include FT_TRUETYPE_TAGS_H

#ifdef _WIN64
#include <fcntl.h>
#include <io.h>
#endif

uint16_t read_uint16(FT_Bytes data)
{
	return (uint16_t)*data << 8 | *(data + 1);
}

uint32_t read_uint32(FT_Bytes data)
{
	return (uint32_t)*data << 24 
		| (uint32_t)*(data + 1) << 16 
		| (uint32_t)*(data + 2) << 8
		| *(data + 3);
}

std::vector<uint16_t> loadLangSys(FT_Bytes data)
{
	uint16_t requiredFeatureIndex = read_uint16(data + 2);
	uint16_t featureIndexCount = read_uint16(data + 4);
	std::vector<uint16_t> featureIndies;
	for(int i = 0; i < featureIndexCount; i++) {
		uint16_t featureIndex = read_uint16(data + 6 + i * 2);
		featureIndies.push_back(featureIndex);
	}
	return featureIndies;
}

int main(int argc, char *argv[])
{
	FT_Library  library;
	FT_Face     face;
	FT_Error    error;

	if(argc < 3) {
		fprintf(stderr, "Usage: %s font_path size\n",argv[0]);
		return 0;
	}

	error = FT_Init_FreeType(&library);
	if(error) {
		fprintf(stderr, "FT_Init_FreeType error %d\n", error);
		return 1;
	}

	error = FT_New_Face(library, argv[1], 0, &face);
	if(error) {
		fprintf(stderr, "FT_New_Face error %d %s\n", error, argv[1]);
		return 1;
	}

#ifdef _WIN64
	_setmode(_fileno(stdin), _O_BINARY);
	_setmode(_fileno(stdout), _O_BINARY);
#endif

	std::map<uint16_t, uint16_t> glyphMap;
	FT_ULong length = 0;
	error = FT_Load_Sfnt_Table(face, TTAG_GSUB, 0, NULL, &length);
	if(!error) {
		std::vector<uint8_t> GSUB_table(length);
		error = FT_Load_Sfnt_Table(face, TTAG_GSUB, 0, GSUB_table.data(), &length);
		if(error) {
			fprintf(stderr, "FT_Load_Sfnt_Table error %d\n", error);
		}
		else {
			if(GSUB_table.size() > 0){
				uint8_t *table = GSUB_table.data();
				uint16_t majorVersion = *(uint16_t *)table;
				uint16_t minorVersion = *((uint16_t *)table + 1);
				uint16_t scriptListOffset = 0;
				uint16_t featureListOffset = 0;
				uint16_t lookupListOffset = 0;
				uint32_t featureVariationsOffset = 0;
				if(majorVersion == 0x0100 && minorVersion == 0x0000) {
					scriptListOffset = read_uint16(table + 4);
					featureListOffset = read_uint16(table + 4 + 2);
					lookupListOffset = read_uint16(table + 4 + 2 * 2);
				}
				if(majorVersion == 0x0100 && minorVersion == 0x0100) {
					scriptListOffset = read_uint16(table + 4);
					featureListOffset = read_uint16(table + 4 + 2);
					lookupListOffset = read_uint16(table + 4 + 2 * 2);
					featureVariationsOffset = read_uint32(table + 4 + 2 * 3);
				}
				std::vector<uint16_t> featureIndies;
				if(scriptListOffset > 0) {
					FT_Bytes ScriptList = table + scriptListOffset;
					uint16_t scriptCount = read_uint16(ScriptList);
					std::map<std::string, uint16_t> ScriptMap;
					uint16_t selectScriptOffset = 0;
					for(int i = 0; i < scriptCount; i++) {
						const char *tag = (const char *)(ScriptList + 2 + (4 + 2) * i);
						uint16_t scriptOffset = read_uint16(ScriptList + 2 + (4 + 2) * i + 4);
						ScriptMap[std::string(tag, 4)] = scriptOffset;
						if(i == 0) {
							selectScriptOffset = scriptOffset;
						}
					}
					if(ScriptMap.count("kana")) {
						selectScriptOffset = ScriptMap["kana"];
					}
					else if(ScriptMap.count("hani")) {
						selectScriptOffset = ScriptMap["hani"];
					}
					else if(ScriptMap.count("DFLT")) {
						selectScriptOffset = ScriptMap["DFLT"];
					}
					else if(ScriptMap.count("latn")) {
						selectScriptOffset = ScriptMap["latn"];
					}
					if(selectScriptOffset > 0){
						FT_Bytes Script = ScriptList + selectScriptOffset;
						uint16_t defaultLangSys = read_uint16(Script);
						if(defaultLangSys > 0) {
							featureIndies = loadLangSys(Script + defaultLangSys);
						}
						uint16_t langSysCount = read_uint16(Script + 2);
						for(int j = 0; j < langSysCount; j++) {
							const char *tag = (const char * )(Script + 4 + (4 + 2) * j);
							uint16_t langSysOffset = read_uint16(Script + 4 + (4 + 2) * j + 4);
							if(strncmp(tag, "JAN ", 4) == 0) {
								featureIndies = loadLangSys(Script + langSysOffset);
							}
						}
					}
				}
				std::vector<int> vertLookupIndies;
				if(featureListOffset > 0) {
					FT_Bytes FeatureList = table + featureListOffset;
					uint16_t featureCount = read_uint16(FeatureList);
					for(uint16_t i = 0; i < featureCount; i++) {
						if(featureIndies.size() == 0 
								|| std::find(featureIndies.begin(), featureIndies.end(), i) != featureIndies.end()) {
							const char *tag = (const char * )(FeatureList + 2 + (4 + 2) * i);
							uint16_t featureOffset = read_uint16(FeatureList + 2 + (4 + 2) * i + 4);
							FT_Bytes Feature = FeatureList + featureOffset;
							if(strncmp(tag, "vert", 4) == 0) {
								uint16_t lookupIndexCount = read_uint16(Feature + 2);
								for(int j = 0; j < lookupIndexCount; j++) {
									uint16_t lookupListIndex = read_uint16(Feature + 4 + j * 2);
									vertLookupIndies.push_back(lookupListIndex);
								}
							}
						}
					}
				}
				if(lookupListOffset > 0) {
					FT_Bytes LookupList = table + lookupListOffset;
					uint16_t lookupCount = read_uint16(LookupList);
					for(int i = 0; i < lookupCount; i++) {
						uint16_t lookupOffset = read_uint16(LookupList + 2 + i * 2);
						FT_Bytes Lookup = LookupList + lookupOffset;
						if(std::find(vertLookupIndies.begin(), vertLookupIndies.end(), i) != vertLookupIndies.end()) {
							uint16_t lookupType = read_uint16(Lookup);
							if(lookupType == 1 || lookupType == 7) {
								uint16_t subTableCount = read_uint16(Lookup + 4);
								for(int j = 0; j < subTableCount; j++) {
									uint16_t subtableOffset = read_uint16(Lookup + 6 + 2 * j);
									FT_Bytes subtable = Lookup + subtableOffset;
									if (lookupType == 7) {
										uint16_t substFormat = read_uint16(subtable);
										uint16_t extensionLookupType = read_uint16(subtable + 2);
										uint32_t extensionOffset = read_uint32(subtable + 4);
										subtable = Lookup + extensionOffset;
										if(extensionLookupType != 1 && substFormat != 1) {
											continue;
										}
									}

									uint16_t substFormat = read_uint16(subtable);
									uint16_t coverageOffset = read_uint16(subtable + 2);
									FT_Bytes coverage = subtable + coverageOffset;
									uint16_t coverageFormat = read_uint16(coverage);
									if(substFormat == 1) {
										uint16_t tmp = read_uint16(subtable + 4);
										short deltaGlyphID = *(short *)&tmp;
										if(coverageFormat == 1) {
											uint16_t glyphCount = read_uint16(coverage + 2);
											for(int g = 0; g < glyphCount; g++) {
												uint16_t glyph = read_uint16(coverage + 4 + g * 2);
												glyphMap[glyph] = glyph + deltaGlyphID;
											}
										}
										else if(coverageFormat == 2) {
											uint16_t rangeCount = read_uint16(coverage + 2);
											for(int r = 0; r < rangeCount; r++) {
												uint16_t startGlyphID = read_uint16(coverage + 4 + r * 6);
												uint16_t endGlyphID = read_uint16(coverage + 4 + r * 6 + 2);
												for(int g = startGlyphID; g <= endGlyphID; g++) {
													glyphMap[g] = g + deltaGlyphID;
												}
											}
										}
									}
									else if(substFormat == 2){
										uint16_t glyphCount2 = read_uint16(subtable + 4);
										if(coverageFormat == 1) {
											uint16_t glyphCount = read_uint16(coverage + 2);
											if(glyphCount2 >= glyphCount) {
												for(int g = 0; g < glyphCount; g++) {
													uint16_t glyph = read_uint16(coverage + 4 + g * 2);
													uint16_t substituteGlyphID = read_uint16(subtable + 6 + g * 2);
													glyphMap[glyph] = substituteGlyphID;
												}
											}
										}
										else if(coverageFormat == 2) {
											uint16_t rangeCount = read_uint16(coverage + 2);
											for(int r = 0; r < rangeCount; r++) {
												uint16_t startGlyphID = read_uint16(coverage + 4 + r * 6);
												uint16_t endGlyphID = read_uint16(coverage + 4 + r * 6 + 2);
												uint16_t startCoverageIndex = read_uint16(coverage + 4 + r * 6 + 4);
												for(int g = startGlyphID; g <= endGlyphID; g++) {
													int CoverageIndex = g - startGlyphID + startCoverageIndex;
													if(CoverageIndex < glyphCount2) {
														uint16_t substituteGlyphID = read_uint16(subtable + 6 + CoverageIndex * 2);
														glyphMap[g] = substituteGlyphID;
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	double  size_f;
	std::istringstream(std::string(argv[2])) >> size_f;
	FT_F26Dot6 size = size_f*64;

	error = FT_Set_Char_Size(
          face,    /* handle to face object           */
          0,       /* char_width in 1/64th of points  */
          size,    /* char_height in 1/64th of points */
          72,      /* horizontal device resolution    */
          72 );    /* vertical device resolution      */
	if(error) {
		fprintf(stderr, "FT_Set_Char_Size error %d\n", error);
		return 1;
	}

	FT_ULong  charcode = 0;
	while(fread(&charcode, 4, 1, stdin) == 1) {
		FT_UInt glyph_index = FT_Get_Char_Index(face, charcode);
		//fprintf(stderr,"charcode %lu %d\n", charcode, glyph_index);
		if(glyph_index == 0) {
			uint32_t rows = 0;
			uint32_t width = 0;
			int32_t boundingWidth = 0;
			int32_t boundingHeight = 0;
			int32_t horiBearingX = 0;
			int32_t horiBearingY = 0;
			int32_t horiAdvance = 0;
			int32_t vertBearingX = 0;
			int32_t vertBearingY = 0;
			int32_t vertAdvance = 0;
			fwrite(&charcode, sizeof(uint32_t), 1, stdout);
			fwrite(&rows, sizeof(uint32_t), 1, stdout);
			fwrite(&width, sizeof(uint32_t), 1, stdout);
			fwrite(&boundingWidth, sizeof(int32_t), 1, stdout);
			fwrite(&boundingHeight, sizeof(int32_t), 1, stdout);
			fwrite(&horiBearingX, sizeof(int32_t), 1, stdout);
			fwrite(&horiBearingY, sizeof(int32_t), 1, stdout);
			fwrite(&horiAdvance, sizeof(int32_t), 1, stdout);
			fflush(stdout);
			continue;
		}

		error = FT_Load_Glyph(
		  face,          /* handle to face object */
		  glyph_index,   /* glyph index           */
		  FT_LOAD_DEFAULT);  /* load flags, see below */
		if(error) {
			fprintf(stderr, "FT_Load_Glyph error %d\n", error);
			return 1;
		}

		error = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
		if(error) {
			fprintf(stderr, "FT_Render_Glyph error %d\n", error);
			return 1;
		}

		FT_GlyphSlot  slot = face->glyph;
		uint32_t rows = slot->bitmap.rows;
		uint32_t width = slot->bitmap.width;
		int32_t boundingWidth = slot->metrics.width;
		int32_t boundingHeight = slot->metrics.height;
		int32_t horiBearingX = slot->metrics.horiBearingX;
		int32_t horiBearingY = slot->metrics.horiBearingY;
		int32_t horiAdvance = slot->metrics.horiAdvance;
		int32_t vertBearingX = slot->metrics.vertBearingX;
		int32_t vertBearingY = slot->metrics.vertBearingY;
		int32_t vertAdvance = slot->metrics.vertAdvance;

		if(rows * width == 0) {
			fwrite(&charcode, sizeof(uint32_t), 1, stdout);
			fwrite(&rows, sizeof(uint32_t), 1, stdout);
			fwrite(&width, sizeof(uint32_t), 1, stdout);
			fwrite(&boundingWidth, sizeof(int32_t), 1, stdout);
			fwrite(&boundingHeight, sizeof(int32_t), 1, stdout);
			fwrite(&horiBearingX, sizeof(int32_t), 1, stdout);
			fwrite(&horiBearingY, sizeof(int32_t), 1, stdout);
			fwrite(&horiAdvance, sizeof(int32_t), 1, stdout);
			fflush(stdout);
			continue;
		}

		fwrite(&charcode, sizeof(uint32_t), 1, stdout);
		fwrite(&rows, sizeof(uint32_t), 1, stdout);
		fwrite(&width, sizeof(uint32_t), 1, stdout);
		fwrite(&boundingWidth, sizeof(int32_t), 1, stdout);
		fwrite(&boundingHeight, sizeof(int32_t), 1, stdout);
		fwrite(&horiBearingX, sizeof(int32_t), 1, stdout);
		fwrite(&horiBearingY, sizeof(int32_t), 1, stdout);
		fwrite(&horiAdvance, sizeof(int32_t), 1, stdout);
		fwrite(slot->bitmap.buffer, sizeof(uint8_t), rows*width, stdout);

		if(glyphMap.count(glyph_index) > 0) {
			error = FT_Load_Glyph(
			face,                    /* handle to face object */
			glyphMap[glyph_index],   /* glyph index           */
			FT_LOAD_DEFAULT);        /* load flags, see below */
			if(error) {
				fprintf(stderr, "FT_Load_Glyph error %d\n", error);
				return 1;
			}

			error = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
			if(error) {
				fprintf(stderr, "FT_Render_Glyph error %d\n", error);
				return 1;
			}

			FT_GlyphSlot  slot2 = face->glyph;
			uint32_t rows2 = slot2->bitmap.rows;
			uint32_t width2 = slot2->bitmap.width;
			int32_t boundingWidth2 = slot2->metrics.width;
			int32_t boundingHeight2 = slot2->metrics.height;
			int32_t vertBearingX2 = slot2->metrics.vertBearingX;
			int32_t vertBearingY2 = slot2->metrics.vertBearingY;
			int32_t vertAdvance2 = slot2->metrics.vertAdvance;

			fwrite(&rows2, sizeof(uint32_t), 1, stdout);
			fwrite(&width2, sizeof(uint32_t), 1, stdout);
			fwrite(&boundingWidth2, sizeof(int32_t), 1, stdout);
			fwrite(&boundingHeight2, sizeof(int32_t), 1, stdout);
			fwrite(&vertBearingX2, sizeof(int32_t), 1, stdout);
			fwrite(&vertBearingY2, sizeof(int32_t), 1, stdout);
			fwrite(&vertAdvance2, sizeof(int32_t), 1, stdout);
			fwrite(slot2->bitmap.buffer, sizeof(uint8_t), rows2*width2, stdout);
		}
		else {
			fwrite(&rows, sizeof(uint32_t), 1, stdout);
			fwrite(&width, sizeof(uint32_t), 1, stdout);
			fwrite(&boundingWidth, sizeof(int32_t), 1, stdout);
			fwrite(&boundingHeight, sizeof(int32_t), 1, stdout);
			fwrite(&vertBearingX, sizeof(int32_t), 1, stdout);
			fwrite(&vertBearingY, sizeof(int32_t), 1, stdout);
			fwrite(&vertAdvance, sizeof(int32_t), 1, stdout);
			fwrite(slot->bitmap.buffer, sizeof(uint8_t), rows*width, stdout);
		}
		fflush(stdout);
	}

	return 0;	
}
