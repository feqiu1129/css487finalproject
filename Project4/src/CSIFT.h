/* CSIFT generates descriptors by stacking SIFT descriptors for the O1/O3 and O2/O3 channels. */
#ifndef __OPENCV_C_SIFT_H__
#define __OPENCV_C_SIFT_H__

#include "OpponentSIFT.h"
using namespace std;
using namespace cv;
#ifdef __cplusplus

namespace cv
{

	class CV_EXPORTS_W CSIFT : public OpponentSIFT
	{
	public:
		CV_WRAP static Ptr<CSIFT> create()
		{
			return makePtr<CSIFT>(CSIFT());
		};
		CV_WRAP explicit CSIFT();
		
		//! returns the descriptor size in floats
		CV_WRAP int descriptorSize() const;

	protected:
		virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;
	};

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
