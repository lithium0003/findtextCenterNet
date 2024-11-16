#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_TRUETYPE_TABLES_H
#include FT_TRUETYPE_TAGS_H

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

	double angle = 0;

	if(argc < 2) {
		fprintf(stderr, "Usage: %s font_path\n", argv[0]);
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

	fprintf(stderr, "font %s\n", argv[1]);

	std::map<uint16_t, uint16_t> vertGlyphMap;
	std::map<uint16_t, std::map<std::string, uint16_t>> ligaGlyphMap;
	FT_ULong length = 0;
	error = FT_Load_Sfnt_Table(face, TTAG_GSUB, 0, NULL, &length);
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
								subtableList.push_back(subtable + extensionOffset);
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
							fprintf(stderr, "vertLookupIndies\n");
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
							fprintf(stderr, "ligaLookupIndies\n");
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
									for(int k = 0; k < ligatureCount; k++) {
										uint16_t ligatureOffset = read_uint16(ligatureSetTable + 2 + k * 2);
										FT_Bytes ligatureTable = ligatureSetTable + ligatureOffset;

										uint16_t ligatureGlyph = read_uint16(ligatureTable);
										uint16_t componentCount = read_uint16(ligatureTable + 2);
										std::ostringstream ss;
										for(int l = 0; l < componentCount - 1; l++) {
											uint16_t componentGlyphID = read_uint16(ligatureTable + 4 + l * 2);
											ss << componentGlyphID << " ";
										}
										subLigature[ss.str()] = ligatureGlyph;
									}
									ligaGlyphMap[convergeGlyph[j]] = subLigature;
								}								
							}
						}
					}
				}
			}
		}
	}

	std::cout << "vertGlyphMap" << std::endl;
	for(const auto& [k,v]: vertGlyphMap) {
		std::cout << k << "->" << v << std::endl;
	}

	std::cout << "ligaGlyphMap" << std::endl;
	for(const auto& [k,vec]: ligaGlyphMap) {
		for(const auto& [s,v]: vec) {
			std::cout << k << " " << s << "->" << v << std::endl;

			error = FT_Load_Glyph(
				face,                    /* handle to face object */
				v,                       /* glyph index           */
				FT_LOAD_DEFAULT);        /* load flags, see below */
			if(error) continue;

			error = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
			if(error) continue;

			FT_GlyphSlot  slot = face->glyph;
			uint32_t rows = slot->bitmap.rows;
			uint32_t width = slot->bitmap.width;

			auto p = slot->bitmap.buffer;
			std::ostringstream ss;
			ss << v << ".txt";
			std::ofstream ofile(ss.str());

			for(int y = 0; y < rows; y++) {
				for(int x = 0; x < width; x++) {
					int value = *p++;
					ofile << value << " ";
				}
				ofile << std::endl;
			}
		}
	}
	return 0;	
}
