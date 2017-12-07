/*
Authors: Nick Huebner, Clark Olson, Siqi Zhang, Sam Hoover
DescriptorType.h
*/

#ifndef DESCRIPTOR_TYPE_H
#define DESCRIPTOR_TYPE_H

#include <vector>
#include <string>
#include <sstream>
#include <opencv2\features2d.hpp>
#include <opencv2/opencv.hpp>
using namespace std;

const char SPACE = ' ';
const char DESCRIPTOR_STRING_DELIM = '+';
const int EOL = -1;

// Enum of the basic descriptor types that have been implemented so far
enum DESC_TYPES {
	_SIFT, 
	_SURF,
	_RGBSIFT,
	_OpponentSIFT,
	_HoNC,
	_HoNC3,
	_HoWH,
	_HoNI,
	_SPIN,
	_CSPIN,
	_RGSIFT, 
	_CSIFT,
	_CHoNI,
	_PSIFT,
	NONE
};


// A struct that holds the type of a descriptor
struct DescriptorType {

	struct Descriptor {
		string name;
		DESC_TYPES type;
	};
	vector<Descriptor> descs;
	bool multiDescriptors;
	string name;

    // Helper to convert from string types to enums types
	static DESC_TYPES getDescriptorType(string code)
    {
        if (code == "SIFT") {
            return _SIFT;
		}
		else if (code == "SURF") {
			return _SURF;
		}
		else if (code == "OpponentSIFT") {
            return _OpponentSIFT;
		} 
		else if (code == "HoNC") {
			return _HoNC;
		}
		else if (code == "HoNC3") {
			return _HoNC3;
		}
		else if (code == "HoWH") {
			return _HoWH;
		}
		else if (code == "HoNI") {
			return _HoNI;
		}
		else if (code == "SPIN") {
			return _SPIN;
		}
		else if (code == "RGBSIFT") {
			return _RGBSIFT;
		}
		else if (code == "RGSIFT") {
			return _RGSIFT;
		}
		else if (code == "CSIFT") {
			return _CSIFT;
		}
		else if (code == "CSPIN") {
			return _CSPIN;
		}
		else if (code == "CHoNI") {
			return _CHoNI;
		}
		else if (code == "PSIFT") {
			return _PSIFT;
		}
		else {
			const int buffersize = 100;
			char error[buffersize] = "Unrecognized type in getDescriptorType: ";
			strcat_s(error, buffersize, code.c_str());
			CV_Error(CV_StsBadArg, error); 
			return NONE;
        }
    }
	//destructor
	~DescriptorType() {}

	//assignment operator
	void operator=(const DescriptorType&copy) {
		this->multiDescriptors = copy.multiDescriptors;
		this->descs = copy.descs;
		this->name = copy.name;
	}

	// Construct a descriptor type from a string code
	DescriptorType(string code) {
		parseDescriptorNames(code);
		multiDescriptors = (descs.size() > 1) ? true : false;
	}


	// Default constructor for a descriptor type nothing
	DescriptorType() : multiDescriptors(false), descs(NULL) {}

	// Construct a descriptor type by explicitly specifying the enums
	//DescriptorType(bool isDouble, DESC_TYPES* descriptors) : multiDescriptors(isDouble), descriptors(descriptors) {}

	void parseDescriptorNames(string names) {
		stringstream sin(names);
		
		char ch = sin.peek();

		while(ch != EOL) {
			string token = "";

			if(ch == DESCRIPTOR_STRING_DELIM) { sin.get(); ch = sin.peek(); }
			while(ch != EOL && ch != DESCRIPTOR_STRING_DELIM) {
				if(ch == ' ') { sin.get(); ch = sin.peek(); }
				else {
					token += sin.get();
					ch = sin.peek();
				}
			}

			Descriptor desc;
			name += (ch == EOL) ? token : (token + "+");
			desc.name = token;
			desc.type = getDescriptorType(token);

			descs.push_back(desc);
		}

	}

};

#endif
