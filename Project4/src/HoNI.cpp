//
///**********************************************************************************************\
//Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
//Below is the original copyright.
//
////    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
////    All rights reserved.
//
//\**********************************************************************************************/

#include "HoNI.h"
using namespace cv;
using namespace xfeatures2d;

// Average standard deviation to normalize greylevels to
static const float AVG_STD_DEV = 64.f;

// constructor
HoNI::HoNI()
{
	// do nothing
}

void HoNI::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const {
	Point pt(cvRound(ptf.x), cvRound(ptf.y));
	float cos_t = cosf(ori*(float)(CV_PI / 180));
	float sin_t = sinf(ori*(float)(CV_PI / 180));
	float bins_per_intensity = n / 255.f;

	float hist_width = VanillaSIFT::SIFT_DESCR_SCL_FCTR * scl;
	int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);

	// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int)sqrt((double)img.cols*img.cols + img.rows*img.rows));
	cos_t /= hist_width;
	sin_t /= hist_width;

	int i;
	int j;
	int k;

	int len = (radius * 2 + 1)*(radius * 2 + 1);
	int histlen = (d + 2)*(d + 2)*(n + 2);
	int rows = img.rows, cols = img.cols;
	AutoBuffer<float> buf(len * 4 + histlen);
	float *RBin = buf, *CBin = RBin + len, *hist = CBin + len, *intensity = hist + len;

	// zeros out the histogram
	for (i = 0; i < d + 2; i++)
	{
		for (j = 0; j < d + 2; j++)
			for (k = 0; k < n + 2; k++)
				hist[(i*(d + 2) + j)*(n + 2) + k] = 0.;
	}

	float ibar = 0, ibar2 = 0;

	for (i = -radius, k = 0; i <= radius; i++)
		for (j = -radius; j <= radius; j++)
		{
			// Calculate sample's histogram array coords rotated relative to ori.
			// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			// r_rot = 1.5) have full weight placed in row 1 after interpolation.
			float c_rot = j * cos_t - i * sin_t;
			float r_rot = j * sin_t + i * cos_t;
			float rbin = r_rot + d / 2 - 0.5f;
			float cbin = c_rot + d / 2 - 0.5f;
			int r = pt.y + i, c = pt.x + j;

			if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
			r > 0 && r < rows - 1 && c > 0 && c < cols - 1)
			{
				RBin[k] = rbin; CBin[k] = cbin;

				// setting the intensity value of each pixel 
				intensity[k] = img.at<VanillaSIFT::sift_wt>(r, c);

				ibar += intensity[k];
				ibar2 += intensity[k] * intensity[k];

				k++;
			}
		}

	len = k;

	ibar = ibar / (float)len;
	ibar2 = ibar2 / (float)len;

	float isig = sqrt(ibar2 - ibar*ibar);

	float bias = 127.5f - ibar;
	float gain = AVG_STD_DEV / isig;

	for (k = 0; k < len; k++)
	{

		intensity[k] = (intensity[k] - ibar)*gain + ibar + bias;

		float rbin = RBin[k], cbin = CBin[k];
		float ibin = (intensity[k]) * bins_per_intensity;

		int r0 = cvFloor(rbin);
		int c0 = cvFloor(cbin);
		int i0 = cvFloor(ibin);

		rbin -= r0;
		cbin -= c0;
		ibin -= i0;
			
		// histogram update using tri-linear interpolation
		float v_r1 = rbin, v_r0 = 1 - v_r1;
		float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
		float v_rco111 = v_rc11*ibin, v_rco110 = v_rc11 - v_rco111;
		float v_rco101 = v_rc10*ibin, v_rco100 = v_rc10 - v_rco101;
		float v_rco011 = v_rc01*ibin, v_rco010 = v_rc01 - v_rco011;
		float v_rco001 = v_rc00*ibin, v_rco000 = v_rc00 - v_rco001;

		int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + i0;
		hist[idx] += v_rco000;
		hist[idx + 1] += v_rco001;
		hist[idx + (n + 2)] += v_rco010;
		hist[idx + (n + 3)] += v_rco011;
		hist[idx + (d + 2)*(n + 2)] += v_rco100;
		hist[idx + (d + 2)*(n + 2) + 1] += v_rco101;
		hist[idx + (d + 3)*(n + 2)] += v_rco110;
		hist[idx + (d + 3)*(n + 2) + 1] += v_rco111;
	}

	// finalize histogram, since the orientation histograms are circular
	for (i = 0; i < d; i++)
		for (j = 0; j < d; j++)
		{
		int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
		hist[idx] += hist[idx + n];
		hist[idx + 1] += hist[idx + n + 1];
		for (k = 0; k < n; k++)
			dst[(i*d + j)*n + k] = hist[idx + k];
		}
	// copy histogram to the descriptor,
	// apply hysteresis thresholding
	// and scale the result, so that it can be easily converted
	// to byte array
	float nrm2 = 0;
	len = d*d*n;
	for (k = 0; k < len; k++)
		nrm2 += dst[k] * dst[k];
	float thr = std::sqrt(nrm2)*VanillaSIFT::SIFT_DESCR_MAG_THR;
	for (i = 0, nrm2 = 0; i < k; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val*val;
	}
	nrm2 = VanillaSIFT::SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);


	for (k = 0; k < len; k++)
	{
		dst[k] =dst[k] * nrm2;
	}
}