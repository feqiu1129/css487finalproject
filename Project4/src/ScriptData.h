//-------------------------------------------------------------------------
// Name: ScriptData.h
// Author: Nick Huebner, Clark Olson, Siqi Zhang, Sam Hoover
// Description: A data structure that stores specifications for the program execution.
//	This structure can only be created from a file with the appropriate parameters.
//-------------------------------------------------------------------------
#ifndef SCRIPTDATA_H
#define SCRIPTDATA_H

#include "ConfigurationManager.h"
#include "DescriptorType.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

static const int MAX_FEATURES = 1000;
static const string DATASETS_FOLDER = "datasets/";
static const string OUTPUT_DIRECTORY = "output/";
static const string SEPARATOR = "/";
static const string TWO_STEPS = "../../";

class DescriptorUtil;

class ScriptData {

public:

	struct ImageSet {
		string name;
		string path;
		int count;
		vector<string> imageNames;
		vector<string> homographyFile;
	};

	struct DataSet {
		DataSet();
		DataSet(string dataset, vector<string> imageset, string proDir, bool resetImgNames, bool isFromConsole);

		string path;
		bool hasUniqueHomographies;
		bool resetImageNames;
		vector<string> imageSetNames;
		ImageSet activeImageSet;
	};

	//indicates data readin succeed/fail 
	bool failed = false;
	bool saveData = false;
	bool drawMatches = false;
	bool isRunningFromConsole;
	bool homographyFlag;

	string projectDirectory;
	DataSet dataset;
    int numberOfDescriptors;
	vector<DescriptorType> descriptorTypes;
    cv::Mat *homographies;
	DescriptorUtil *descriptorUtil;
	DESC_TYPES featureExtractor;
	string featureExtractorText;
	int descriptorTableSize;

	ScriptData();
	ScriptData(ConfigurationManager configs);
	ScriptData(char *args[]);
	ScriptData(const ScriptData& copy);
	~ScriptData();

	void run();
	
private:
	void setNumberOfImages(int number);
	void setImageNames(vector<string> names);
	void resetImageNamesToCurrentImageset();
	void setNumberOfDescriptors(int number);
	void setDescriptorTypes(vector<string> descs);
	void setHomographies(vector<string> homographies);
	void setHasUniqueHomographies(bool unique);
	void buildHomographiesMatrix();
	void setFeatureExtractor(string extractor);
	void outputSpecs();

	// run helper functions
	void runAllImageSets();
	void runActiveImageSet();
	void initTable(Mat*** table);
	void initDescriptors(Mat **descriptors);
	void computeKeypoints(vector<KeyPoint> *kpts, Mat *images, string* imageNames);
	void writeKeypointsToFile(vector<cv::KeyPoint> *kpts, string* imageNames);
	void computeDescriptors(Mat **descriptors, Mat*** table, vector<KeyPoint> *kpts, Mat *images, string* imageNames);
	Mat computeDescriptor(int descIndex, int imagesetIndex, Mat*** table, vector<KeyPoint> *kpts, Mat *images);
	void writeDescriptorToFile(Mat **descriptors, string* imageNames, int descIndex);
	void performMatching(Mat **descriptors, Mat*** table, vector<KeyPoint> *kpts, Mat *images, int descIndex);
	void freeMemory(Mat*** table, Mat **descriptors, vector<cv::KeyPoint> *kpts, Mat *images, string *imageNames);

	string removeFileExtension(string str);

};


#endif
