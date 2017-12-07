/* ConfigurationManager.h
 * by: Sam Hoover
 */

#ifndef CONFIGURATIONMANAGER_H
#define CONFIGURATIONMANAGER_H

//#include "ScriptData.h"
#include <algorithm>
#include <io.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
using namespace std;


static const string DEFAULT_CONFIG_FILE_PATH = "config/config.txt";
static const string DEFAULT_CONFIG_DIRECTORY = "config/";
static const string TWO_UP = "../../";
static const string PROJECT_DIR_INDICATOR_TOKEN = "/46x";
static const string TRUE_TOKEN = "true";
static const string EMPTY_STRING = "";

static const char ID_DELIM = ':';
static const char FILE_SEPARATOR = '/';

enum CONFIG {
	DATASET,
	IMAGESET,
	IMAGES,
	HOMOGRAPHIES,
	DESCRIPTOR,
	EXTRACTOR,
	SAVE,
	DISPLAY
};


class ConfigurationManager {
public:

	struct Configuration {
	public:
		const char EOL = -1;
		const char NEW_PARAMETER_IDENTIFIER = ',';
	
		string identifier;
		vector<string> specs;
		void create(stringstream &config);
	private:
		string getIdentifier(stringstream &sin);
	};
	
	const string DATASET_IDENTIFIER = "dataset";
	const string IMAGESET_IDENTIFIER = "imageset";
	const string IMAGES_IDENTIFIER = "images";
	const string HOMOGRAPHIES_IDENTIFIER = "homographies";
	const string DESCRIPTOR_IDENTIFIER = "descriptors";
	const string EXTRACTOR_IDENTIFIER = "extractor";
	const string SAVE_IDENTIFIER = "save";
	const string DISPLAY_IDENTIFIER = "display";

	const string OXFORD_DATASET = "oxford";


	string projectDirectory;
	string dataset;
	vector<string> imageset;
	vector<string> images;
	vector<string> homographies;
	vector<string> descriptors;
	string extractor;
	bool save = false;
	bool display = false;
	bool uniqueHomographies = false;
	bool resetImageNames = false;
	bool isRunningFromConsole;

	ConfigurationManager();
	ConfigurationManager(string configPath);
	ConfigurationManager(ConfigurationManager &cm);

	ConfigurationManager& operator=(const ConfigurationManager &cm);

	void init();
	bool isValid() const { return(valid); }
private:
	bool valid = false;
	string configFile;

	vector<Configuration> readConfigurationFile();
	void setConfiguration(Configuration config, int type);
	void validate();
	void setProjectDirectory();
	void setIsRunningFromConsole();
};




#endif