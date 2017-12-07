/* HoNI generates descriptors by histogramming normalized greyvalues into a 4x4x8 grid. */
#ifndef __OPENCV_HONI_H__
#define __OPENCV_HONI_H__

#include "VanillaSIFT.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;
using namespace hal;

#ifdef __cplusplus

/*!
HoNI implementation.
*/

class CV_EXPORTS_W HoNI : public VanillaSIFT {
public:

	CV_WRAP static Ptr<HoNI> create() {
			return makePtr<HoNI>(HoNI());
		};

	CV_WRAP explicit HoNI();

protected:

	virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;

};

#endif /* __cplusplus */

#endif