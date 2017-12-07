/* SPIN generates descriptors using a circular histogram of 8 radii and 16 (normalized) greylevels */
#ifndef SPIN_H
#define SPIN_H

#include "VanillaSIFT.h"
#include <cmath>

using namespace std;
using namespace cv;
using namespace xfeatures2d;
using namespace hal;

#ifdef __cplusplus

/*!
SPIN implementation.
*/

class CV_EXPORTS_W SPIN : public VanillaSIFT {

	const int NUM_INTENSITY_BINS = 16;
	const int NUM_DISTANCE_BINS = 8;

public:

	CV_WRAP static Ptr<SPIN> create() {
			return makePtr<SPIN>(SPIN());
		};

	CV_WRAP explicit SPIN();

protected:

	virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;
	virtual void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints, Mat& descriptors, int nOctaveLayers, int firstOctave) const;
	CV_WRAP virtual int descriptorSize() const;
};

#endif /* __cplusplus */

#endif