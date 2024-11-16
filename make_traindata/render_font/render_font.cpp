#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sstream>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TRUETYPE_TABLES_H
#include FT_TRUETYPE_TAGS_H
#include FT_SYNTHESIS_H
#ifdef _WIN64
#include <fcntl.h>
#include <io.h>
#endif

bool italic = false;
bool bold = false;

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

void load_convert(FT_Face &face, std::map<uint16_t, uint16_t> &vertGlyphMap, std::map<uint16_t, std::pair<uint16_t, std::map<std::string, uint16_t>>> &ligaGlyphMap)
{
	FT_ULong length = 0;
	FT_Error error = FT_Load_Sfnt_Table(face, TTAG_GSUB, 0, NULL, &length);
	if(!error) {
		std::vector<uint8_t> GSUB_table(length);
		error = FT_Load_Sfnt_Table(face, TTAG_GSUB, 0, GSUB_table.data(), &length);
		if(error) {
			fprintf(stderr, "FT_Load_Sfnt_Table error %d\n", error);
		}
		else {
			std::map<std::string, std::map<std::string, std::vector<uint16_t>>> langFeatureIndies;
			std::vector<std::pair<std::string, std::vector<uint16_t>>> featureLookupListIndices;
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
				if(scriptListOffset > 0) {
					FT_Bytes ScriptList = table + scriptListOffset;
					uint16_t scriptCount = read_uint16(ScriptList);
					std::map<std::string, uint16_t> ScriptMap;
					for(int i = 0; i < scriptCount; i++) {
						const char *tag = (const char *)(ScriptList + 2 + (4 + 2) * i);
						uint16_t scriptOffset = read_uint16(ScriptList + 2 + (4 + 2) * i + 4);
						ScriptMap[std::string(tag, 4)] = scriptOffset;
					}
					for (const auto& [k, v] : ScriptMap) {
						FT_Bytes Script = ScriptList + v;
						uint16_t defaultLangSys = read_uint16(Script);
						std::map<std::string, std::vector<uint16_t>> featureIndies;
						if(defaultLangSys > 0) {
							featureIndies["default"] = loadLangSys(Script + defaultLangSys);
						}
						uint16_t langSysCount = read_uint16(Script + 2);
						for(int j = 0; j < langSysCount; j++) {
							const char *tag = (const char * )(Script + 4 + (4 + 2) * j);
							uint16_t langSysOffset = read_uint16(Script + 4 + (4 + 2) * j + 4);
							featureIndies[std::string(tag, 4)] = loadLangSys(Script + langSysOffset);
						}
						langFeatureIndies[k] = featureIndies;
					}
				}
				if(featureListOffset > 0) {
					FT_Bytes FeatureList = table + featureListOffset;
					uint16_t featureCount = read_uint16(FeatureList);
					for(uint16_t i = 0; i < featureCount; i++) {
						const char *tag = (const char * )(FeatureList + 2 + (4 + 2) * i);
						uint16_t featureOffset = read_uint16(FeatureList + 2 + (4 + 2) * i + 4);
						FT_Bytes Feature = FeatureList + featureOffset;

						uint16_t lookupIndexCount = read_uint16(Feature + 2);
						std::vector<uint16_t> lookupListIndices;
						for(int j = 0; j < lookupIndexCount; j++) {
							uint16_t lookupListIndex = read_uint16(Feature + 4 + j * 2);
							lookupListIndices.push_back(lookupListIndex);
						}
						featureLookupListIndices.emplace_back(std::make_pair(std::string(tag, 4), lookupListIndices));
					}
				}

				std::vector<uint16_t> vertLookupIndies;
				if(langFeatureIndies.count("kana")) {
					if(langFeatureIndies["kana"].count("JAN ")) {
						for(auto idx: langFeatureIndies["kana"]["JAN "]) {
							if(featureLookupListIndices[idx].first == "vert") {
								vertLookupIndies = featureLookupListIndices[idx].second;
							}
						}
					}
					else {
						for(auto idx: langFeatureIndies["kana"]["default"]) {
							if(featureLookupListIndices[idx].first == "vert") {
								vertLookupIndies = featureLookupListIndices[idx].second;
							}
						}
					}
				}
				else if(langFeatureIndies.count("hani")) {
					if(langFeatureIndies["hani"].count("JAN ")) {
						for(auto idx: langFeatureIndies["hani"]["JAN "]) {
							if(featureLookupListIndices[idx].first == "vert") {
								vertLookupIndies = featureLookupListIndices[idx].second;
							}
						}
					}
					else {
						for(auto idx: langFeatureIndies["hani"]["default"]) {
							if(featureLookupListIndices[idx].first == "vert") {
								vertLookupIndies = featureLookupListIndices[idx].second;
							}
						}
					}
				}
				else if(langFeatureIndies.count("DFLT")) {
					if(langFeatureIndies["DFLT"].count("JAN ")) {
						for(auto idx: langFeatureIndies["DFLT"]["JAN "]) {
							if(featureLookupListIndices[idx].first == "vert") {
								vertLookupIndies = featureLookupListIndices[idx].second;
							}
						}
					}
					else {
						for(auto idx: langFeatureIndies["DFLT"]["default"]) {
							if(featureLookupListIndices[idx].first == "vert") {
								vertLookupIndies = featureLookupListIndices[idx].second;
							}
						}
					}
				}
				else if(langFeatureIndies.count("latn")) {
					if(langFeatureIndies["latn"].count("JAN ")) {
						for(auto idx: langFeatureIndies["latn"]["JAN "]) {
							if(featureLookupListIndices[idx].first == "vert") {
								vertLookupIndies = featureLookupListIndices[idx].second;
							}
						}
					}
					else {
						for(auto idx: langFeatureIndies["latn"]["default"]) {
							if(featureLookupListIndices[idx].first == "vert") {
								vertLookupIndies = featureLookupListIndices[idx].second;
							}
						}
					}
				}

				std::vector<uint16_t> ligaLookupIndies;
				if(langFeatureIndies.count("DFLT")) {
					for(auto idx: langFeatureIndies["DFLT"]["default"]) {
						if(featureLookupListIndices[idx].first == "liga") {
							ligaLookupIndies = featureLookupListIndices[idx].second;
						}
					}
				}
				else if(langFeatureIndies.count("latn")) {
					for(auto idx: langFeatureIndies["latn"]["default"]) {
						if(featureLookupListIndices[idx].first == "liga") {
							ligaLookupIndies = featureLookupListIndices[idx].second;
						}
					}
				}

				if(lookupListOffset > 0) {
					FT_Bytes LookupList = table + lookupListOffset;
					uint16_t lookupCount = read_uint16(LookupList);
					for(int i = 0; i < lookupCount; i++) {
						uint16_t lookupOffset = read_uint16(LookupList + 2 + i * 2);
						FT_Bytes Lookup = LookupList + lookupOffset;
						uint16_t lookupType = read_uint16(Lookup);

						std::vector<FT_Bytes> subtableList;
						if(lookupType == 7) {
							uint16_t subTableCount = read_uint16(Lookup + 4);
							for(int j = 0; j < subTableCount; j++) {
								uint16_t subtableOffset = read_uint16(Lookup + 6 + 2 * j);
								FT_Bytes subtable = Lookup + subtableOffset;
								uint16_t substFormat = read_uint16(subtable);
								uint16_t extensionLookupType = read_uint16(subtable + 2);
								uint32_t extensionOffset = read_uint32(subtable + 4);
								subtable = Lookup + extensionOffset;
								subtableList.push_back(subtable);
								lookupType = extensionLookupType;
							}
						}
						else {
							uint16_t subTableCount = read_uint16(Lookup + 4);
							for(int j = 0; j < subTableCount; j++) {
								uint16_t subtableOffset = read_uint16(Lookup + 6 + 2 * j);
								FT_Bytes subtable = Lookup + subtableOffset;
								subtableList.push_back(subtable);
							}
						}

						if(lookupType == 1 && std::find(vertLookupIndies.begin(), vertLookupIndies.end(), i) != vertLookupIndies.end()) {
							for(const auto& subtable: subtableList) {
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
											vertGlyphMap[glyph] = glyph + deltaGlyphID;
										}
									}
									else if(coverageFormat == 2) {
										uint16_t rangeCount = read_uint16(coverage + 2);
										for(int r = 0; r < rangeCount; r++) {
											uint16_t startGlyphID = read_uint16(coverage + 4 + r * 6);
											uint16_t endGlyphID = read_uint16(coverage + 4 + r * 6 + 2);
											for(int g = startGlyphID; g <= endGlyphID; g++) {
												vertGlyphMap[g] = g + deltaGlyphID;
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
												vertGlyphMap[glyph] = substituteGlyphID;
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
													vertGlyphMap[g] = substituteGlyphID;
												}
											}
										}
									}
								}
							}
						}
						if(lookupType == 4 && std::find(ligaLookupIndies.begin(), ligaLookupIndies.end(), i) != ligaLookupIndies.end()) {
							for(const auto& subtable: subtableList) {
								uint16_t substFormat = read_uint16(subtable);
								uint16_t coverageOffset = read_uint16(subtable + 2);
								FT_Bytes coverage = subtable + coverageOffset;
								uint16_t coverageFormat = read_uint16(coverage);
								std::vector<uint16_t> convergeGlyph;
								if(coverageFormat == 1) {
									uint16_t glyphCount = read_uint16(coverage + 2);
									for(int g = 0; g < glyphCount; g++) {
										uint16_t glyph = read_uint16(coverage + 4 + g * 2);
										convergeGlyph.push_back(glyph);
									}
								}
								else if(coverageFormat == 2) {
									uint16_t rangeCount = read_uint16(coverage + 2);
									for(int r = 0; r < rangeCount; r++) {
										uint16_t startGlyphID = read_uint16(coverage + 4 + r * 6);
										uint16_t endGlyphID = read_uint16(coverage + 4 + r * 6 + 2);
										uint16_t startCoverageIndex = read_uint16(coverage + 4 + r * 6 + 4);
										for(int g = startGlyphID; g <= endGlyphID; g++) {
											uint16_t glyph = g + startCoverageIndex;
											convergeGlyph.push_back(glyph);
										}
									}
								}
								uint16_t ligatureSetCount = read_uint16(subtable + 4);
								for(int j = 0; j < ligatureSetCount; j++) {
									if(j >= convergeGlyph.size()) break;
									uint16_t ligatureSetOffset = read_uint16(subtable + 6 + j * 2);
									FT_Bytes ligatureSetTable = subtable + ligatureSetOffset;

									uint16_t ligatureCount = read_uint16(ligatureSetTable);
									std::map<std::string, uint16_t> subLigature;
									uint16_t max_componentCount = 0;
									for(int k = 0; k < ligatureCount; k++) {
										uint16_t ligatureOffset = read_uint16(ligatureSetTable + 2 + k * 2);
										FT_Bytes ligatureTable = ligatureSetTable + ligatureOffset;

										uint16_t ligatureGlyph = read_uint16(ligatureTable);
										uint16_t componentCount = read_uint16(ligatureTable + 2);
										max_componentCount = std::max(max_componentCount, componentCount);
										std::ostringstream ss;
										for(int l = 0; l < componentCount - 1; l++) {
											uint16_t componentGlyphID = read_uint16(ligatureTable + 4 + l * 2);
											ss << componentGlyphID << " ";
										}
										subLigature[ss.str()] = ligatureGlyph;
									}
									ligaGlyphMap[convergeGlyph[j]] = std::make_pair(max_componentCount, subLigature);
								}								
							}
						}
					}
				}
			}
		}
	}
}

int output_glyph(FT_Face &face, uint32_t liga_count, FT_UInt glyph_index, std::map<uint16_t, uint16_t> &vertGlyphMap)
{
	FT_Error error = FT_Load_Glyph(
		face,              /* handle to face object */
		glyph_index,       /* glyph index           */
		FT_LOAD_DEFAULT);  /* load flags, see below */
	if(error) {
		fprintf(stderr, "FT_Load_Glyph error %d\n", error);
		return 1;
	}

	if (italic)
		FT_GlyphSlot_Oblique(face->glyph );  //斜体にする
	if (bold)
		FT_GlyphSlot_Embolden(face->glyph );//太字にする

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

	fwrite(&liga_count, sizeof(uint32_t), 1, stdout);
	if(rows * width == 0) {
		fwrite(&rows, sizeof(uint32_t), 1, stdout);
		fwrite(&width, sizeof(uint32_t), 1, stdout);
		fwrite(&boundingWidth, sizeof(int32_t), 1, stdout);
		fwrite(&boundingHeight, sizeof(int32_t), 1, stdout);
		fwrite(&horiBearingX, sizeof(int32_t), 1, stdout);
		fwrite(&horiBearingY, sizeof(int32_t), 1, stdout);
		fwrite(&horiAdvance, sizeof(int32_t), 1, stdout);
		return 0;
	}

	fwrite(&rows, sizeof(uint32_t), 1, stdout);
	fwrite(&width, sizeof(uint32_t), 1, stdout);
	fwrite(&boundingWidth, sizeof(int32_t), 1, stdout);
	fwrite(&boundingHeight, sizeof(int32_t), 1, stdout);
	fwrite(&horiBearingX, sizeof(int32_t), 1, stdout);
	fwrite(&horiBearingY, sizeof(int32_t), 1, stdout);
	fwrite(&horiAdvance, sizeof(int32_t), 1, stdout);
	fwrite(slot->bitmap.buffer, sizeof(uint8_t), rows*width, stdout);

	if(liga_count == 1 && vertGlyphMap.count(glyph_index) > 0) {
		FT_UInt vert_glyph_index = vertGlyphMap[glyph_index];
		error = FT_Load_Glyph(
			face,                    /* handle to face object */
			vert_glyph_index,        /* glyph index           */
			FT_LOAD_DEFAULT);        /* load flags, see below */
		if(error) {
			fprintf(stderr, "FT_Load_Glyph error %d\n", error);
			return 1;
		}

		if (italic)
			FT_GlyphSlot_Oblique(face->glyph );  //斜体にする
		if (bold)
			FT_GlyphSlot_Embolden(face->glyph );//太字にする

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
	return 0;
}

int render_glyph(FT_Face &face, const std::vector<FT_UInt> glyphindex_list, std::map<uint16_t, uint16_t> &vertGlyphMap, std::map<uint16_t, std::pair<uint16_t, std::map<std::string, uint16_t>>> &ligaGlyphMap)
{
	FT_Error    error;
	int i = 0;
	while(i < glyphindex_list.size()) {
		FT_UInt glyph_index = glyphindex_list[i];
		if(glyph_index == 0) {
			uint32_t liga_count = 1;
			uint32_t rows = 0;
			uint32_t width = 0;
			int32_t boundingWidth = 0;
			int32_t boundingHeight = 0;
			int32_t bearingX = 0;
			int32_t bearingY = 0;
			int32_t advance = 0;
			fwrite(&liga_count, sizeof(uint32_t), 1, stdout);
			fwrite(&rows, sizeof(uint32_t), 1, stdout);
			fwrite(&width, sizeof(uint32_t), 1, stdout);
			fwrite(&boundingWidth, sizeof(int32_t), 1, stdout);
			fwrite(&boundingHeight, sizeof(int32_t), 1, stdout);
			fwrite(&bearingX, sizeof(int32_t), 1, stdout);
			fwrite(&bearingY, sizeof(int32_t), 1, stdout);
			fwrite(&advance, sizeof(int32_t), 1, stdout);
		}		
		else if(ligaGlyphMap.count(glyph_index) == 0) {
			if(output_glyph(face, 1, glyph_index, vertGlyphMap) != 0) {
				return 1;
			}
		}
		else {
			int count = ligaGlyphMap[glyph_index].first;
			for(int j = count; j > 1; j--) {
				if(i + j <= glyphindex_list.size()) {
					std::ostringstream ss;
					for(int c = i+1; c < i+j; c++) {
						ss << glyphindex_list[c] << " ";
					}
					if(ligaGlyphMap[glyph_index].second.count(ss.str()) > 0) {
						FT_UInt ligature_glyph = ligaGlyphMap[glyph_index].second[ss.str()];
						if(output_glyph(face, j, ligature_glyph, vertGlyphMap) != 0){
							return 1;
						}
						i += j;
						goto next;
					}
				}
			}
			if(output_glyph(face, 1, glyph_index, vertGlyphMap) != 0) {
				return 1;
			}
		}
		i++;
		next:
		;
	}
	return 0;
}

int main(int argc, char *argv[])
{
	FT_Library  library;
	FT_Face     face;
	FT_Error    error;

	double angle = 0;

	if(argc < 4) {
		fprintf(stderr, "Usage: %s font_path size type\n", argv[0]);
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

	std::map<uint16_t, uint16_t> vertGlyphMap;
	std::map<uint16_t, std::pair<uint16_t, std::map<std::string, uint16_t>>> ligaGlyphMap;
	load_convert(face, vertGlyphMap, ligaGlyphMap);

	double  size_f;
	std::istringstream(std::string(argv[2])) >> size_f;
	FT_F26Dot6 size = size_f*64;

	int t = 0;
	std::istringstream(std::string(argv[3])) >> t;
	italic = (t & 1) == 1;
	bold = (t & 2) == 2;

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

	std::vector<FT_UInt> glyphindex_list;
	FT_ULong  charcode = 0;
	while(fread(&charcode, 4, 1, stdin) == 1) {
		if(charcode == 0) {
			if(render_glyph(face, glyphindex_list, vertGlyphMap, ligaGlyphMap) != 0){
				return 1;
			}
			fflush(stdout);
			glyphindex_list.clear();
		}
		else {
			FT_UInt glyph_index = FT_Get_Char_Index(face, charcode);
			glyphindex_list.push_back(glyph_index);		
		}
	}
	if(glyphindex_list.size() > 0) {
		if(render_glyph(face, glyphindex_list, vertGlyphMap, ligaGlyphMap) != 0) {
			return 1;
		};
	}
	fflush(stdout);
	
	return 0;	
}
