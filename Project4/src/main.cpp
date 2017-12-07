#include "ConfigurationManager.h"
#include "ScriptData.h"
#include <iostream>
using namespace std;


int main(int argc, char *argv[])
{
	ConfigurationManager configs;

	if(argc > 1) { configs = ConfigurationManager(argv[1]); }

	configs.init();
	if(configs.isValid()) {
		ScriptData data(configs);
		data.run();
	}
	return(0);
}