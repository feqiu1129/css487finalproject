/*
Authors: Nick Huebner, Clark Olson

This class provides utilities for computing key points and different types of descriptors.
*/

#include "DescriptorUtil.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv::xfeatures2d;

// Constructor, initializes parameters to be used for the keypoint detectors and descriptor extractors
DescriptorUtil::DescriptorUtil()
{
}

// Desctructor
DescriptorUtil::~DescriptorUtil()
{
}

// Detect features in an image using the SIFT feature detector. The keyPoints parameter will contain the key points detected
void DescriptorUtil::detectFeatures(const Mat& img, vector<KeyPoint> &keyPoints, ScriptData data)
{

	/*		// Code to use if ORB detector is added
	Ptr<ORB> p = ORB::create(1000);
	p->detect(img, keyPoints);
	for (int i = 0; i < keyPoints.size(); i++) {
		keyPoints[i].size /= 2;
		keyPoints[i].octave = 0;
		float cur = keyPoints[i].size;
		while (cur > 16.0) {
			cur /= 2.0;
			keyPoints[i].octave++;
		}
	}
	//for (int i = 0; i < keyPoints.size(); i++)
	//	printf("%4d: %2d %5.1f\n", i, keyPoints[i].octave, keyPoints[i].size);
	return;
	*/
	
	/*		// Code to use if BRISK detector is added
	Ptr<BRISK> p = BRISK::create();
	p->detect(img, keyPoints);
	for (int i = 0; i < keyPoints.size(); i++) {
		keyPoints[i].size /= 1;
		keyPoints[i].octave = 0;
		float cur = keyPoints[i].size;
		while (cur > 16.0) {
			cur /= 2.0;
			keyPoints[i].octave++;
		}
	}
	return;
	*/

	// Note MSER, A/KAZE, GFTT, Fast, Agast are all unsuitable, since no angle information is computed (could be done after the fact, though)

    DESC_TYPES type = data.featureExtractor;
	// SIFT detector
	if (type == _SIFT) {
		Ptr<VanillaSIFT> sift = VanillaSIFT::create();
		(*sift)(img, noArray(), keyPoints, noArray(), false);
	}
	// SURF detector
	else if (type == _SURF) {
		Ptr<SURF> surf = SURF::create();
		surf->detect(img, keyPoints);
	}
	// Color histogram SIFT
	else if (type == _HoNC) {
		Ptr<HoNC> honc = HoNC::create();
		(*honc)(img, noArray(), keyPoints, noArray(), false);
	}
	else
	{
		CV_Error(CV_StsBadArg, "Unrecognized type in detectFeatures");
	}
}

// Reads key points from a file
vector<KeyPoint> DescriptorUtil::readKeyPoints(string filePath, string imgName)
{
    vector<KeyPoint> keyPoints;
    // Read features from file
    FileStorage fs(filePath, FileStorage::READ);
    read(fs[imgName], keyPoints);
    fs.release();
    return keyPoints;
}

// Writes key points to a file (.xml or .yml)
void DescriptorUtil::writeKeyPoints(vector<KeyPoint> *kpts, string *imgNames, int numImgs, string filename)
{
    // Write keypoints out to file
    FileStorage fs(filename, FileStorage::WRITE);
    for (int i = 0; i < numImgs; ++i) 
	{
        size_t dotLocation = imgNames[i].rfind('.');
        write(fs, imgNames[i].substr(0, dotLocation), kpts[i]);
    }
    fs.release();
}

void DescriptorUtil::normalizeDescriptors(Mat &descriptors) {
	float nrm2 = 0;
	for (int j = 0; j < descriptors.rows; j++) {
		int len = descriptors.cols;
		
		for (int k = 0; k < len; k++)
			nrm2 += powf(descriptors.at<float>(j, k), 2);

		const float SIFT_INT_DESCR_FCTR = 512.f;
		nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

		for (int k = 0; k < len; k++)
		{
			descriptors.at<float>(j, k) = (descriptors.at<float>(j, k) * nrm2);
		}
	}
}

// Computes the descriptors of a specified type for an image, given a set of keypoints
Mat DescriptorUtil::computeDescriptors(Mat& img, vector<KeyPoint> &keypoints, DESC_TYPES type)
{
    Mat descriptors;
    vector<KeyPoint> kpts(keypoints.begin(), keypoints.end());
    

    // Lowe's SIFT Descriptor: descriptor size = 128
    if (type == _SIFT) {
		Ptr<VanillaSIFT> sift = VanillaSIFT::create();
		sift->compute(img, kpts, descriptors);
    }
	// SURF Descriptor: descriptor size = 64
	else if (type == _SURF) {
		// Ptr<SURF> surf = SURF::create(100.0, 4, 3, true, false);	//extended SURF doesn't seem to improve things!
		// SURF appears to use scale a bit differently. Increase scale some to improve results.
		for (int i = 0; i < kpts.size(); i++) kpts[i].size *= 2.0;
		Ptr<SURF> surf = SURF::create();
		surf->compute(img, kpts, descriptors);
		normalizeDescriptors(descriptors);	// Want all methods to have same vector length
	}
	// RGB SIFT: descriptor size = 384
	else if (type == _RGBSIFT) {
		Ptr<RGBSIFT> rgbsift = RGBSIFT::create();
		rgbsift->compute(img, kpts, descriptors);
	}
	// Opponent SIFT: descriptor size = 384
	else if (type == _OpponentSIFT) {
		Ptr<OpponentSIFT>oppenentsift = OpponentSIFT::create();
		oppenentsift->compute(img, kpts, descriptors);
	}
	// Color histogram SIFT : descriptor size = 128
	else if (type == _HoNC) {
		Ptr<HoNC> honc = HoNC::create();
		honc->compute(img, kpts, descriptors);
	}
	// Color histogram SIFT with 3 x 3 x 3 color hist: descriptor size = 432
	else if (type == _HoNC3) {
		Ptr<HoNC3> honc3 = HoNC3::create();
		honc3->compute(img, kpts, descriptors);
	}
	// Hue weighted by saturation SIFT : descriptor size = 128
	else if (type == _HoWH) {
		Ptr<HoWH> howh = HoWH::create();
		howh->compute(img, kpts, descriptors);
	}
	// Gresycale texture SIFT: descriptor size = 128
	else if (type == _HoNI){
		Ptr<HoNI> honi = HoNI::create();
		honi->compute(img, kpts, descriptors);
	}
	// RGBIntensity: descriptor size = 384
	else if (type == _CHoNI) {
		Ptr<CHoNI> choni = CHoNI::create();
		choni->compute(img, kpts, descriptors);
	}
	// rgSIFT: descriptor size = 256 (384, but last 128 are all zero with current implementation)
	else if (type == _RGSIFT) {
		Ptr<RGSIFT> rgsift = RGSIFT::create();
		rgsift->compute(img, kpts, descriptors);
	}
	// CSIFT: descriptor size = 256
	else if (type == _CSIFT) {
		Ptr<CSIFT> csift = CSIFT::create();
		csift->compute(img, kpts, descriptors);
	}
	// SPIN: descriptor size = 128
	else if (type == _SPIN){
		Ptr<SPIN> spin = SPIN::create();
		spin->compute(img, kpts, descriptors);
	}
	// CSPIN: descriptor size = 384
	else if (type == _CSPIN){
		Ptr<CSPIN> cspin = CSPIN::create();
		cspin->compute(img, kpts, descriptors);
	}
	else if (type == _PSIFT) {
		Ptr<PSIFT> psift = PSIFT::create();
		psift->compute(img, kpts, descriptors);
	}
	else if (type == NONE) { }
	else {
		CV_Error(CV_StsBadArg, "Unrecognized type in computeDescriptors");
	}
    return descriptors;
}

// Merge multiple descriptors. There should be an equal number of descriptors in the matrices
// descriptorArray contains all descriptors
// num is the number of descriptors
Mat DescriptorUtil::mergeDescriptors(Mat* descriptorArray, int num)
{
	//calculate the size of merged descriptor
	int newrows = descriptorArray[0].rows, newcols = 0;
	for (int i = 0; i < num; i++)
		newcols += descriptorArray[i].cols;
	//create an empty descriptor
	Mat descriptors(0, newcols, descriptorArray[0].type());
	for (int i = 0; i < descriptorArray[0].rows; ++i)
	{
		vector<float> mergedRow;
		for (int descriptorIndex = 0; descriptorIndex < num; descriptorIndex++)
		{
			//only merge rows when the descripor has equavalent row numbers
			if (descriptorArray[0].rows == descriptorArray[descriptorIndex].rows)
			{
				// Get the corresponding rows
				Mat m1 = descriptorArray[descriptorIndex].row(i);
				for (int j = 0; j < descriptorArray[descriptorIndex].cols; ++j)
					mergedRow.push_back(m1.at<float>(j));
			}
		}
		// Create a matrix
		Mat x(mergedRow);
		Mat y = x.t();
		descriptors.push_back(y);
		mergedRow.clear();
	}
    return descriptors;
}

// Writes descriptors to a file (.xml or .yml)
void DescriptorUtil::writeDescriptors(Mat *&descriptors, string *imgNames, int numImgs, string filename)
{
    // Write descriptors out to file
    FileStorage fs(filename, FileStorage::WRITE);
    for (int i = 0; i < numImgs; ++i) 
	{
        size_t dotLocation = imgNames[i].rfind('.');
        write(fs, imgNames[i].substr(0, dotLocation), descriptors[i]);
    }
    fs.release();
}

// Matches descriptors from two different images, evaluates the matches using the provided homography, and writes the results out to a file
void DescriptorUtil::match(const Mat &descr1, Mat &descr2, 
					  const vector<KeyPoint> &kpts1, const vector<KeyPoint> &kpts2, const Mat &img1, const Mat &img2, 
					  const Mat &homography, const string outFilename, bool drawMatches)
{
    // matching descriptors
	srand(1);   // ensure random numbers used in matcher are repeatable
    // FlannBasedMatcher matcher;
	BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descr1, descr2, matches);
    int totalMatches = (int)matches.size();
    sort(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2) {
        return m1.distance < m2.distance;
    });
    bool *correct = new bool[totalMatches];
    memset(correct, 0, totalMatches);

	bool debug = false;
    int outBounds = 0;
	int matchesToDisplay = 200;
    vector<char> matchesMask( totalMatches, 0 );
    for (int i = 0; i < totalMatches; ++i) {
		if (i < matchesToDisplay) matchesMask[i] = 1; else matchesMask[i] = 0;
        Point p1 = kpts1[matches[i].queryIdx].pt; // image 1 point
        Point p2 = kpts2[matches[i].trainIdx].pt; // image 2 point
        Mat p1Mat = (Mat_<double>(3, 1) << (double)p1.x, (double)p1.y, 1.0);
        Mat p2Mat = (Mat_<double>(3, 1) << (double)p2.x, (double)p2.y, 1.0);

		// checking reverse direction, too!
		Mat p1Mat2 = (Mat_<double>(3, 1) << (double)p1.x, (double)p1.y, 1.0);
		Mat p2Mat2 = (Mat_<double>(3, 1) << (double)p2.x, (double)p2.y, 1.0);
		

        // if (norm(p2 - H * p1 / H.z)) < 2 * p2.size
        p1Mat = homography * p1Mat;
        p1Mat /= p1Mat.at<double>(2, 0);
        double x = p1Mat.at<double>(0, 0);
        double y = p1Mat.at<double>(1, 0);
        // Check for out of bounds
        if (x < 0 || x > img2.cols || y < 0 || y > img2.rows) {
            outBounds++;
        } else {
            Mat res = p2Mat - p1Mat;
			p2Mat2 = homography.inv() * p2Mat2;
			p2Mat2 /= p2Mat2.at<double>(2, 0);
			Mat res2 = p2Mat2 - p1Mat2;
			if (norm(res) < kpts2[matches[i].trainIdx].size  && norm(res2) < kpts1[matches[i].queryIdx].size) {
				correct[i] = true;
			}

			if (debug && i < matchesToDisplay) printf("%3d: %4d %4d - %4d %4d: ", i, p1.x, p1.y, p2.x, p2.y);
			if (debug && i < matchesToDisplay) {
				if (correct[i]) printf("good\n"); else printf("bad\n");
			}
        }
    }

    ofstream outFile(outFilename.c_str());
	outFile << totalMatches << "\t" << (kpts1.size() - outBounds) << endl;

	// Find precision and recall values by traversing match list and considering matches in order of distance
	int numCorrect = 0; 
	outFile << 0 << "\t" << 0 << endl;
	for (int i = 0; i < totalMatches; i++) {
		if (i > 0 && matches[i].distance < matches[i - 1].distance) {
			CV_Error(CV_BadOrder, "Matches must be sorted according to quality.");
		}

		if (correct[i]) {
			++numCorrect;
		}
		outFile << numCorrect << "\t" << i + 1 << endl;
	}

    outFile.close();

	cout << ">> Output results to: " << outFilename << endl;
	printf("\t%3d correct matches\n", numCorrect);

    // drawing the results
    if (drawMatches) {
        namedWindow("Match Results", 1);
        Mat img_matches;

        cv::drawMatches(img1, kpts1, img2, kpts2, matches, img_matches,
                        Scalar::all(-1), Scalar::all(-1), matchesMask);
        imshow("Match Results", img_matches);
        imwrite("matches.jpg", img_matches);
        waitKey(0);
        destroyWindow("Match Results");
    }

    delete [] correct;
}
