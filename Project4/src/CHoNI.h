/* CHoNI generates descriptors by stacking HoNI descriptors for the R, G, and B channels. */
#ifndef __OPENCV_CHoNI_H__
#define __OPENCV_CHoNI_H__

#include "RGBSIFT.h"
using namespace std;
using namespace cv;
#ifdef __cplusplus

namespace cv
{

	class CV_EXPORTS_W CHoNI : public RGBSIFT 
	{
	public:
		CV_WRAP static Ptr<CHoNI> create()
		{
			return makePtr<CHoNI>(CHoNI());
		};
		CV_WRAP explicit CHoNI();
		
		//------------------------------------descriptorSize()---------------------------------
		// ! returns the descriptor size in floats
		//Precondition: None
		//Postcondition: the descriptor size is returned in floats
		//-------------------------------------------------------------------------------------
		CV_WRAP virtual int descriptorSize() const;

	protected:
		virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;

	private:
		void calculateIntensity(int band, float bins_per_intensity, int d, int n, float *hist, int radius, float cos_t, float sin_t, Point pt, int rows, int cols, float *CBin, float *RBin, const  Mat& img, int histlen, float *intensity) const;
		void normalizeHistogram(float *dst, int d, int n) const;
	};

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
