/* RG SIFT generates descriptors by stacking SIFT descriptors for the R/(R+G+B) and G/(R+G+B) channels. */
#ifndef __OPENCV_RG_SIFT_H__
#define __OPENCV_RG_SIFT_H__

#include "RGBSIFT.h"
using namespace std;
using namespace cv;
#ifdef __cplusplus

namespace cv
{

	class CV_EXPORTS_W RGSIFT : public RGBSIFT
	{
	public:
		CV_WRAP static Ptr<RGSIFT> create()
		{
			return makePtr<RGSIFT>(RGSIFT());
		};
		CV_WRAP explicit RGSIFT();
		
		//! returns the descriptor size in floats
		CV_WRAP int descriptorSize() const;
		
	protected:
		//virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;
		void operator()(InputArray _image, InputArray _mask,
			vector<KeyPoint>& keypoints,
			OutputArray _descriptors,
			bool useProvidedKeypoints) const;
		void convertBGRImage(Mat& bgrImage) const;
		void normalizeHistogram(float *dst, int d, int n) const;
	};

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
