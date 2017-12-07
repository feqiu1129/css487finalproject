#include "ScriptData.h"
#include "DescriptorUtil.h"
#include <opencv2/opencv.hpp>


ScriptData::ScriptData() {}


ScriptData::ScriptData(ConfigurationManager configs) {
	if(configs.isValid()) {
		descriptorTableSize = (int)DESC_TYPES::NONE; // this assumes that NONE is the last entry in the DESC_TYPES enum
		projectDirectory = configs.projectDirectory;
		isRunningFromConsole = configs.isRunningFromConsole;
		saveData = configs.save;
		drawMatches = configs.display;
		dataset = DataSet(configs.dataset, configs.imageset, projectDirectory, configs.resetImageNames, isRunningFromConsole);
		setNumberOfImages((int)configs.images.size());
		setImageNames(configs.images);
		setNumberOfDescriptors((int)configs.descriptors.size());
		setDescriptorTypes(configs.descriptors);
		setFeatureExtractor(configs.extractor); 
		outputSpecs();
		setHasUniqueHomographies(configs.uniqueHomographies);
		setHomographies(configs.homographies);	
	}
}



//------------------------------------ScriptData()-------------------------------------
// copy constructor
//Precondition: copy is appropriately set
//Postcondition: a deep copy is made
//-------------------------------------------------------------------------------------
ScriptData::ScriptData(const ScriptData& copy) {
	if(&copy != this) {
		
		// copy DataSet
		this->dataset = DataSet();
		this->dataset.path = copy.dataset.path;
		this->dataset.hasUniqueHomographies = copy.dataset.hasUniqueHomographies;
		this->dataset.imageSetNames = copy.dataset.imageSetNames;
		
		// copy ImageSet
		this->dataset.activeImageSet.name = copy.dataset.activeImageSet.name;
		this->dataset.activeImageSet.path = copy.dataset.activeImageSet.path;
		this->dataset.activeImageSet.count = copy.dataset.activeImageSet.count;
		this->dataset.activeImageSet.homographyFile = copy.dataset.activeImageSet.homographyFile;
		this->dataset.activeImageSet.imageNames = copy.dataset.activeImageSet.imageNames;

		this->failed = copy.failed;
		this->numberOfDescriptors = copy.numberOfDescriptors;
		this->homographyFlag = copy.homographyFlag;
		this->saveData = copy.saveData;
		this->featureExtractor = copy.featureExtractor;
		this->featureExtractorText = copy.featureExtractorText;
		this->descriptorTypes = copy.descriptorTypes;

		this->homographies = new cv::Mat[dataset.activeImageSet.count - 1];
		for (int i = 0; i < dataset.activeImageSet.count - 1; i++)
		{
			this->homographies[i] = copy.homographies[i];
		}
	}
}

// Clean up allocated memory
ScriptData::~ScriptData()
{
    if (!failed) {
        delete [] homographies;
    }
}

void ScriptData::run() {
	if(dataset.imageSetNames.size() > 1) { runAllImageSets(); } 
	else { runActiveImageSet(); }
}

//------------------------------------runSingleImageSet()--------------------------------------------
//compute descripter / descriptors for one image set
//Precondition: the following parameters must be correclty defined.
//parameters:
	//data: script data
	//*argv[]: command arguments
	//imgIndex: index of the image set
//useProvidedKeypoints: bool indicating whether using provided keypoints
//Postcondition: the descripter/descriptors is/are computed output for certain image
//-------------------------------------------------------------------------------------
void ScriptData::runActiveImageSet()
{
	//allocate space for descriptor pointer hashtable, for one image set
	Mat*** descriptorTable = new Mat**[descriptorTableSize];
	Mat *images = new Mat[dataset.activeImageSet.count];
	// point a string array to the start of the imageNames vector for use in desc util (vectors guarantee contiguous storage)
	string* imageNames = &dataset.activeImageSet.imageNames[0];
	vector<KeyPoint> *kpts = new vector<KeyPoint>[dataset.activeImageSet.count];
	Mat **descriptors = new Mat*[numberOfDescriptors];
	
	initTable(descriptorTable);
	initDescriptors(descriptors);
	computeKeypoints(kpts, images, imageNames);
	computeDescriptors(descriptors, descriptorTable, kpts, images, imageNames);
	freeMemory(descriptorTable, descriptors, kpts, images, imageNames);
}


void ScriptData::runAllImageSets() {

	if (!failed) {
		// we are already set to the first imageset, with the homography created, so we can run
		runActiveImageSet();

		for (int i = 1; i < dataset.imageSetNames.size(); i++) {

			dataset.activeImageSet.path = dataset.path + dataset.imageSetNames[i] + SEPARATOR;
			dataset.activeImageSet.name = dataset.imageSetNames[i];

			if(dataset.resetImageNames) { resetImageNamesToCurrentImageset(); }

			outputSpecs();
			buildHomographiesMatrix();

			runActiveImageSet();
		}
	}
}


void ScriptData::initTable(Mat*** table) {
	
	for (int i = 0; i < descriptorTableSize; i++)
	{
		//row is descriptors type, column is image index
		table[i] = new Mat*[dataset.activeImageSet.count];

		for(int j = 0; j < dataset.activeImageSet.count; j++) {
			table[i][j] = NULL;
		}
	}
}


void ScriptData::initDescriptors(Mat **descriptors) {
	
	for (int i = 0; i < numberOfDescriptors; ++i) {
		descriptors[i] = new Mat[dataset.activeImageSet.count];
	}
}

void ScriptData::computeKeypoints(vector<KeyPoint> *kpts, Mat *images, string* imageNames) {
	// Load images and compute keypoints for each image
	for (int i = 0; i < dataset.activeImageSet.count; ++i) {
		images[i] = imread((dataset.activeImageSet.path + dataset.activeImageSet.imageNames[i]));

		cout << ">> Computing keypoints for " << dataset.activeImageSet.imageNames[i] << "..." << endl;

		// Load from file or detect new features
		ScriptData temp(*this);
		descriptorUtil->detectFeatures(images[i], kpts[i], temp);
		KeyPointsFilter::retainBest(kpts[i], MAX_FEATURES);
	}

	cout << ">> Finished computing all keypoints" << endl;

	// Save keypoints if save flag is set
	if(saveData) { writeKeypointsToFile(kpts, imageNames); }
}


void ScriptData::writeKeypointsToFile(vector<KeyPoint> *kpts, string* imageNames) {
		stringstream keyPs;
		string outputDir = (isRunningFromConsole) ? TWO_STEPS + projectDirectory + OUTPUT_DIRECTORY : OUTPUT_DIRECTORY;

		keyPs << outputDir << "kpts_images=";

		for(int i = 0; i < dataset.activeImageSet.imageNames.size(); i++) {
			keyPs << removeFileExtension(dataset.activeImageSet.imageNames[i]);
			if(i != dataset.activeImageSet.imageNames.size() - 1) { keyPs << "&"; }
		}
		keyPs << ".xml";

		cout << ">> Saving keypoints to: " << keyPs.str() << endl;

		descriptorUtil->writeKeyPoints(kpts, imageNames, dataset.activeImageSet.count, keyPs.str());
}

void ScriptData::computeDescriptors(Mat **descriptors, Mat*** table, vector<KeyPoint> *kpts, Mat *images, string* imageNames) {
	// Compute descriptors

	for(int i = 0; i < numberOfDescriptors; ++i) {
		for(int j = 0; j < dataset.activeImageSet.count; ++j) {
			// Inner array of descriptor matrices contains only one type of descriptor
			descriptors[i][j] = computeDescriptor(i, j, table, kpts, images);
		}
		
		// Save descriptors if save flag is set
		if(saveData) { writeDescriptorToFile(descriptors, imageNames, i); }
		double t, tf = getTickFrequency();
		t = (double)getTickCount();
		performMatching(descriptors, table, kpts, images, i);
		t = (double)getTickCount() - t;
		printf("perform matching time: %g\n", t*1000. / tf);
	}
}


Mat ScriptData::computeDescriptor(int descIndex, int imagesetIndex, Mat*** table, vector<KeyPoint> *kpts, Mat *images) {
	Mat* descriptorArray = new Mat[descriptorTypes[descIndex].descs.size()];

	cout << ">> Computing " << descriptorTypes[descIndex].name << " descriptor for " << dataset.activeImageSet.imageNames[imagesetIndex] << endl;

	for(int k = 0; k < descriptorTypes[descIndex].descs.size(); k++) {

		int type = (int)descriptorTypes[descIndex].descs[k].type;

		// compute descriptor if not yet computed
		if (table[type][imagesetIndex] == NULL) {
			descriptorArray[k] = descriptorUtil->computeDescriptors(images[imagesetIndex], kpts[imagesetIndex], descriptorTypes[descIndex].descs[k].type);
			table[type][imagesetIndex] = new Mat(descriptorArray[k]);
		} else { // descriptor has been computed
			descriptorArray[k] = Mat(*table[type][imagesetIndex]);
		}
	}

	if(descriptorTypes[descIndex].descs.size() > 1) {
		return(descriptorUtil->mergeDescriptors(descriptorArray, (int)descriptorTypes[descIndex].descs.size()));
	}
	return(descriptorArray[0]);
}

void ScriptData::writeDescriptorToFile(Mat **descriptors, string* imageNames, int descIndex) {
	stringstream descriptorFilePath;
	string outputDir = (isRunningFromConsole) ? TWO_STEPS + projectDirectory + OUTPUT_DIRECTORY : OUTPUT_DIRECTORY;

	descriptorFilePath << outputDir << "descriptors=";

	for(int i = 0; i < descriptorTypes.size(); i++) {
		descriptorFilePath << descriptorTypes[i].name;
		if(i != descriptorTypes.size() - 1) { descriptorFilePath << "&"; }
	}
	descriptorFilePath << "_%%_images=";

	for(int i = 0; i < dataset.activeImageSet.imageNames.size(); i++) {
		descriptorFilePath << removeFileExtension(dataset.activeImageSet.imageNames[i]);
		if(i != dataset.activeImageSet.imageNames.size() - 1) { descriptorFilePath << "&"; }
	}		
	descriptorFilePath << ".xml";

	cout << ">> Saving descriptors to: " << descriptorFilePath.str() << endl;

	descriptorUtil->writeDescriptors(descriptors[descIndex], imageNames, dataset.activeImageSet.count, descriptorFilePath.str());
}


void ScriptData::performMatching(Mat **descriptors, Mat*** table, vector<KeyPoint> *kpts, Mat *images, int descIndex) {

	if (homographyFlag) {
		for (int j = 0; j < dataset.activeImageSet.count - 1; ++j) {
			stringstream outFilename;
			
			string comparedImages = removeFileExtension(dataset.activeImageSet.imageNames[0]) + "-" + removeFileExtension(dataset.activeImageSet.imageNames[j + 1]);
			
			string outputDir = (isRunningFromConsole) ? TWO_STEPS + projectDirectory + OUTPUT_DIRECTORY : OUTPUT_DIRECTORY;

			outFilename << outputDir << descriptorTypes[descIndex].name << "_" << dataset.activeImageSet.name << "_" << comparedImages << ".txt";

			descriptorUtil->match(descriptors[descIndex][0], descriptors[descIndex][j + 1], kpts[0], kpts[j + 1], images[0], images[j + 1], homographies[j], outFilename.str(), drawMatches);
		}
	}
}


void ScriptData::freeMemory(Mat*** table, Mat **descriptors, vector<cv::KeyPoint> *kpts, Mat *images, string *imageNames) {
	imageNames = nullptr;
	delete imageNames;
	
	delete[] images;
	delete[] kpts;

	for (int i = 0; i < numberOfDescriptors; ++i) {
		delete[] descriptors[i];
	}
	delete[] descriptors;

	//deallocate the hashtable, backward
	for (int index = descriptorTableSize - 1; index >=0 ; index--) {
		for (int img = dataset.activeImageSet.count - 1; img >= 0; img--) {
			if (table[index][img] != NULL) { delete table[index][img]; }
		}
		delete[] table[index];
	}
	delete[] table;
}


void ScriptData::setNumberOfImages(int number) {
	dataset.activeImageSet.count = number;
}


void ScriptData::setImageNames(vector<string> names) {
	dataset.activeImageSet.imageNames = names;
}


void ScriptData::resetImageNamesToCurrentImageset() {
	for(int i = 0; i < dataset.activeImageSet.imageNames.size(); i++) {
		for(int j = 0; j < dataset.activeImageSet.imageNames[i].size(); j++) {
			if(dataset.activeImageSet.imageNames[i][j] == '_') {
				dataset.activeImageSet.imageNames[i] = dataset.activeImageSet.name + dataset.activeImageSet.imageNames[i].substr(j, dataset.activeImageSet.imageNames[i].size());
				break;
			}
		}
	}
}


void ScriptData::setNumberOfDescriptors(int number) {
	numberOfDescriptors = number;
}


void ScriptData::setDescriptorTypes(vector<string> descs) {
	descriptorTypes = vector<DescriptorType>(numberOfDescriptors);

    for (int i = 0; i < numberOfDescriptors; ++i) {
		descriptorTypes[i] = DescriptorType(descs[i]);	
    }
}


void::ScriptData::setHasUniqueHomographies(bool unique) {
	dataset.hasUniqueHomographies = unique;
}


void ScriptData::setHomographies(vector<string> homographies) {
	dataset.activeImageSet.homographyFile = homographies;
	for(int i = 0; i < dataset.activeImageSet.count - 1; i++) {
		//homographyFlag = (dataset.activeImageSet.homographyFile[i] != "") ? true : false;	
		homographyFlag = (dataset.activeImageSet.homographyFile[(dataset.hasUniqueHomographies) ? i : 0] != "") ? true : false;
	}
	buildHomographiesMatrix();
}


void ScriptData::buildHomographiesMatrix() {
	homographies = new cv::Mat[dataset.activeImageSet.count - 1];

	if (homographyFlag) {
		for (int i = 0; i < dataset.activeImageSet.count - 1; i++) {
			int homographyIndex = (dataset.hasUniqueHomographies) ? i : 0;

			// Read and construct the homography matrix for each image pair
			cout << ">> Homography file: " << dataset.activeImageSet.homographyFile[homographyIndex] << endl;
            double tmpArray[9];

			string homographyPath = (dataset.hasUniqueHomographies) ? dataset.activeImageSet.path : dataset.path;

			ifstream hFile(homographyPath + dataset.activeImageSet.homographyFile[homographyIndex]);

			if (!hFile.is_open()) {
				cout << "Unable to open homography file." << endl;
				cout << "  path: " << homographyPath + dataset.activeImageSet.homographyFile[homographyIndex] << endl;
				failed = true;
				return;
			}
            for (int j = 0; j < 9; ++j) {
                hFile >> tmpArray[j];
            }
			printf(">> Homography: %6.1f %6.1f %6.1f\n", tmpArray[0], tmpArray[1], tmpArray[2]);
			printf(">> Homography: %6.1f %6.1f %6.1f\n", tmpArray[3], tmpArray[4], tmpArray[5]);
			printf(">> Homography: %6.1f %6.1f %6.1f\n", tmpArray[6], tmpArray[7], tmpArray[8]);

            homographies[i] = (cv::Mat_<double>(3,3) << tmpArray[0], tmpArray[1], tmpArray[2], tmpArray[3], tmpArray[4], tmpArray[5], tmpArray[6], tmpArray[7], tmpArray[8]);
	   }
    }
}


void ScriptData::setFeatureExtractor(string extractor) {
	featureExtractorText = extractor;

	if (extractor == "SURF") {
		featureExtractor = _SURF;
	} else if (extractor == "HoNC") {
		featureExtractor = _HoNC;
	} else if (extractor == "SIFT") {
		featureExtractor = _SIFT;
	} else {
		CV_Error(CV_StsBadArg, "Unrecognized extractor type in ScriptData");
	}
}


void ScriptData::outputSpecs() {
	// output, move to method
	cout << ">> Image path: " << dataset.activeImageSet.path << endl;
	for (int i = 0; i < dataset.activeImageSet.count; i++) {
		cout << ">> Image " << i << ": " << dataset.activeImageSet.imageNames[i] << endl;
	}
	cout << ">> Detector: " << featureExtractorText << endl;
	cout << ">> Num descriptors: " << numberOfDescriptors << endl;

    // Get the descriptor types
    for (int i = 0; i < numberOfDescriptors; ++i) {
		cout << ">> Descriptor " << i << ": " << descriptorTypes[i].name << endl;
    }
}


ScriptData::DataSet::DataSet() {}


ScriptData::DataSet::DataSet(string dataset, vector<string> imgset, string proDir, bool resetImgNames, bool isFromConsole) {
	if(isFromConsole) { path = TWO_STEPS + proDir + DATASETS_FOLDER + dataset + SEPARATOR; } 
	else { path = DATASETS_FOLDER + dataset + SEPARATOR; }
	imageSetNames = imgset;
	activeImageSet.path = path + imgset[0] + SEPARATOR;
	activeImageSet.name = imgset[0];
	resetImageNames = resetImgNames;
}


string ScriptData::removeFileExtension(string str) {
	stringstream sin(str);
	string token = "";
	char ch = sin.peek();

	while(ch != EOL && ch != '.') {
		token += sin.get();
		ch = sin.peek();
	}
	return(token);
}
