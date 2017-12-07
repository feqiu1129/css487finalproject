/* CSPIN generates descriptors by stacking SPIN descriptors for the R, G, and B channels. */
#ifndef __OPENCV_COLORSPIN_SIFT_H__
#define __OPENCV_COLORSPIN_SIFT_H__

#include "RGBSIFT.h"
using namespace std;
using namespace cv;
#ifdef __cplusplus

namespace cv
{

	class CV_EXPORTS_W CSPIN : public RGBSIFT {

		const int NUM_INTENSITY_BINS = 16;
		const int NUM_DISTANCE_BINS = 8;

	public:
		CV_WRAP static Ptr<CSPIN> create()
		{
			return makePtr<CSPIN>(CSPIN());
		};

		CV_WRAP explicit CSPIN();
		void CSPIN::normalizeHistogram(float *dst, int d, int n) const;

	protected:
		virtual void calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const;
		virtual void calcBand(const Mat& img, float *dst, int band, float *intensity, float *dist, Point pt, int radius, int len, int intensity_bins, int distance_bins, float bins_per_distance, float bins_per_intensity, float alpha, float beta) const;
		virtual void calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints, Mat& descriptors, int nOctaveLayers, int firstOctave) const;
		CV_WRAP virtual int descriptorSize() const;
	};

} /* namespace cv */

#endif /* __cplusplus */

#endif

/* End of file. */
