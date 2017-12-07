/* RG SIFT generates descriptors by stacking SIFT descriptors for the R/(R+G+B) and G/(R+G+B) channels. */
#ifndef __OPENCV_P_SIFT_H__
#define __OPENCV_P_SIFT_H__

#include "RGBSIFT.h"
using namespace std;
using namespace cv;
#ifdef __cplusplus

namespace cv
{

	class CV_EXPORTS_W PSIFT : public RGBSIFT
	{
	public:
		CV_WRAP static Ptr<PSIFT> create()
		{
			return makePtr<PSIFT>(PSIFT());
		};
		CV_WRAP explicit PSIFT();

	protected:
		void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;
		void normalizePoolHist(float *dst, int d, int n) const;
	};

} /* namespace cv */

#endif /* __cplusplus */

#endif

  /* End of file. */