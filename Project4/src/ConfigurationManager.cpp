/* ConfigurationManager.cpp
 * by: Sam Hoover
 */
#include "ConfigurationManager.h"
#include <windows.h>

ConfigurationManager::ConfigurationManager() : configFile(DEFAULT_CONFIG_FILE_PATH) {}
ConfigurationManager::ConfigurationManager(string config) : configFile(config) {}

ConfigurationManager::ConfigurationManager(ConfigurationManager &cm) {
	projectDirectory = cm.projectDirectory;
	isRunningFromConsole = cm.isRunningFromConsole;
	dataset = cm.dataset;
	imageset = cm.imageset;
	images = cm.images;
	homographies = cm.homographies;
	descriptors = cm.descriptors;
	extractor = cm.extractor;
	save = cm.save;
	display = cm.display;
	uniqueHomographies = cm.uniqueHomographies;
	resetImageNames = cm.resetImageNames;
	valid = cm.valid;
	configFile = cm.configFile;
}


ConfigurationManager& ConfigurationManager::operator=(const ConfigurationManager &cm) {
	if(this != &cm) {
		projectDirectory = cm.projectDirectory;
		isRunningFromConsole = cm.isRunningFromConsole;
		dataset = cm.dataset;
		imageset = cm.imageset;
		images = cm.images;
		homographies = cm.homographies;
		descriptors = cm.descriptors;
		extractor = cm.extractor;
		save = cm.save;
		display = cm.display;
		uniqueHomographies = cm.uniqueHomographies;
		resetImageNames = cm.resetImageNames;
		valid = cm.valid;
		configFile = cm.configFile;
	}
	return(*this);
}


void ConfigurationManager::Configuration::create(stringstream &config) {
	vector<string> tokens;

	identifier = getIdentifier(config);

	char ch = config.peek();

	while(ch != EOL) {
		string token = "";

		if(ch == ',') { config.get(); ch = config.peek(); }
		else {
			while(ch != EOL && ch != ',') {
				if(ch == ' ') { config.get(); ch = config.peek(); }
				else {
					token += config.get();
					ch = config.peek();
				}
			}

			tokens.push_back(token);
		}
	}

	specs = tokens;
}


string ConfigurationManager::Configuration::getIdentifier(stringstream& sin) {
	string identifier = "";
	
	char ch = sin.peek();

	while(ch != EOL && ch != ID_DELIM) {
		identifier += sin.get();
		ch = sin.peek();
	}

	if(ch == ID_DELIM) { sin.get(); }

	return(identifier);
}


void ConfigurationManager::init() {
	
	setIsRunningFromConsole();

	setProjectDirectory();

	vector<Configuration> configs = readConfigurationFile();
	
	for(int i = 0; i < configs.size(); i++) {
		setConfiguration(configs[i], i);
	}

	validate();
}


vector<ConfigurationManager::Configuration> ConfigurationManager::readConfigurationFile() {
	fstream infoFile;
	string line;
	vector<string> configTokens;
	vector<Configuration> configs;

	string path = (isRunningFromConsole) ? (TWO_UP + projectDirectory + DEFAULT_CONFIG_DIRECTORY + configFile) : DEFAULT_CONFIG_FILE_PATH;

	infoFile.open(path);

	while(infoFile.good()) {
		getline(infoFile, line);

		if(line != EMPTY_STRING) {
			Configuration config;
			config.create(stringstream(line));
			configs.push_back(config);
		}	
	}

	infoFile.close();

	return(configs);
}


void ConfigurationManager::setConfiguration(Configuration config, int type) {
	switch(type) {
		case CONFIG::DATASET:

			if(config.identifier == DATASET_IDENTIFIER && config.specs.size() > 0) {
				// only one dataset can be run at a time, so we only care about the first entry
				dataset = config.specs[0];
			}
		
			if(dataset == OXFORD_DATASET) { uniqueHomographies = true; }
			else { resetImageNames = true; }

			break;

		case CONFIG::IMAGESET:

			if(config.identifier == IMAGESET_IDENTIFIER && config.specs.size() > 0) {
				imageset = config.specs;
			}

			break;

		case CONFIG::IMAGES:

			if(config.identifier == IMAGES_IDENTIFIER && config.specs.size() >= 2) {
				images = config.specs;
			}

			break;

		case CONFIG::HOMOGRAPHIES:

			if(config.identifier == HOMOGRAPHIES_IDENTIFIER && config.specs.size() > 0) {
				homographies = config.specs;
			}
			
			break;

		case CONFIG::DESCRIPTOR:

			if(config.identifier == DESCRIPTOR_IDENTIFIER && config.specs.size() > 0) {
				descriptors = config.specs;
			}
			
			break;

		case CONFIG::EXTRACTOR:

			if(config.identifier == EXTRACTOR_IDENTIFIER && config.specs.size() > 0) {
				// only one feature extractor can be set, so we only care about the first entry
				extractor = config.specs[0];
			}
			
			break;

		case CONFIG::SAVE:

			if(config.identifier == DISPLAY_IDENTIFIER && config.specs.size() > 0) {
				save = (config.specs[0] == TRUE_TOKEN) ? true : false;
			}

			break;

		case CONFIG::DISPLAY:

			if(config.identifier == DISPLAY_IDENTIFIER && config.specs.size() > 0) {
				display = (config.specs[0] == TRUE_TOKEN) ? true : false;
			}

			break;

		default:
			cout << "There was an error setting a configuration" << endl;
			valid = false;
	}
}


void ConfigurationManager::validate() {

	if(projectDirectory == EMPTY_STRING) { valid = false; return; }
	if(dataset == EMPTY_STRING) { valid = false; return; }
	if(imageset.empty()) { valid = false; return; }
	if(images.size() < 2) { valid = false; return; }
	if(homographies.empty()) { valid = false; return; }
	if(descriptors.empty()) { valid = false; return; }
	if(extractor == EMPTY_STRING) { valid = false; return; }

	// if we have unique homography files, the images count must be exactly one greater than homographies count
	if(uniqueHomographies && images.size() != homographies.size() + 1) { valid = false; return; }

	valid = true;
}


void ConfigurationManager::setProjectDirectory() {
	string token = "";
	string dirName = "";

	// get the current working directory
	char result[MAX_PATH];
	string cwd = string(result, GetModuleFileName(NULL, result, MAX_PATH));
	
	// ensure that the path is using unix file separators
	replace(cwd.begin(), cwd.end(), '\\', '/');

	// extract the project directory name from the current working directory
	for(int i = (int)(cwd.length() - 1); i > -1; i--) {
		if(cwd[i] == FILE_SEPARATOR) {
			if(token == PROJECT_DIR_INDICATOR_TOKEN) {
				int j = i - 1;
				while(cwd[j] != FILE_SEPARATOR) {
					j--;
				}
				dirName = cwd.substr(j + 1, i - j);
				break;
			}
			token = "";
		}
		token += cwd[i];
	}

	projectDirectory = dirName;

}


void ConfigurationManager::setIsRunningFromConsole() {
	// if the debugger is present, then we are inside Visual Studio, otherwise we are being run from a console
	if(IsDebuggerPresent()) { isRunningFromConsole = false; } 
	else { isRunningFromConsole = true; }
}

