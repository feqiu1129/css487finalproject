
/**********************************************************************************************\
Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

\**********************************************************************************************/


#include "CHoNI.h"

namespace cv
{
	//------------------------------------descriptorSize()---------------------------------
	// ! returns the descriptor size in floats
	//Precondition: None
	//Postcondition: the descriptor size is returned in floats
	//-------------------------------------------------------------------------------------
	int CHoNI::descriptorSize() const {
		return 3 * SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
	}

	void CHoNI::normalizeHistogram(float *dst, int d, int n) const {
		float nrm1sqr = 0, nrm2sqr = 0, nrm3sqr = 0;
		int len = d*d*n;
		for (int k = 0; k < len; k++) {
			nrm1sqr += dst[k] * dst[k];
			nrm2sqr += dst[k + len] * dst[k + len];
			nrm3sqr += dst[k + 2 * len] * dst[k + 2 * len];
		}
		float thr1 = std::sqrt(nrm1sqr)*SIFT_DESCR_MAG_THR;
		float thr2 = std::sqrt(nrm2sqr)*SIFT_DESCR_MAG_THR;
		float thr3 = std::sqrt(nrm3sqr)*SIFT_DESCR_MAG_THR;
		nrm1sqr = nrm2sqr = nrm3sqr = 0;
		for (int k = 0; k < len; k++)
		{
			float val = std::min(dst[k], thr1);
			dst[k] = val;
			nrm1sqr += val*val;
			val = std::min(dst[k + len], thr2);
			dst[k + len] = val;
			nrm2sqr += val*val;
			val = std::min(dst[k + 2 * len], thr3);
			dst[k + 2 * len] = val;
			nrm3sqr += val*val;
		}

		// Factor of three added below to make vector lengths comparable with SIFT
		nrm1sqr = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(3 * nrm1sqr), FLT_EPSILON);
		nrm2sqr = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(3 * nrm2sqr), FLT_EPSILON);
		nrm3sqr = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(3 * nrm3sqr), FLT_EPSILON);
		for (int k = 0; k < len; k++)
		{
			dst[k] = (dst[k] * nrm1sqr);
			dst[k + len] = (dst[k + len] * nrm2sqr);
			dst[k + 2 * len] = (dst[k + 2 * len] * nrm3sqr);
		}

	}


//-------------------------------------------------------------------------------------
	//img: color image
	//ptf: keypoint
	//ori: angle(degree) of the keypoint relative to the coordinates, clockwise
	//scl: radius of meaningful neighborhood around the keypoint 
	//d: newsift descr_width, 4 in this case
	//n: SIFT_descr_hist_bins, 8 in this case
	//dst: descriptor array to pass in
	//changes: 1. img now is a color image
	void CHoNI::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const {
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
		AutoBuffer<float> buf(len * 7 + 3 * histlen);
		float *RBin = buf, *CBin = RBin + len, *hist1 = CBin + len, *hist2 = hist1 + histlen, *hist3 = hist2 + histlen, *red = hist3 + histlen, *green = red + len, *blue = green + len;

		calculateIntensity(0, bins_per_intensity, d, n, hist1, radius, cos_t, sin_t, pt, rows, cols, CBin, RBin, img, histlen, blue);
		calculateIntensity(1, bins_per_intensity, d, n, hist2, radius, cos_t, sin_t, pt, rows, cols, CBin, RBin, img, histlen, green);
		calculateIntensity(2, bins_per_intensity, d, n, hist3, radius, cos_t, sin_t, pt, rows, cols, CBin, RBin, img, histlen, red);

		// finalize histogram, since the orientation histograms are circular
		for (i = 0; i < d; i++)
			for (j = 0; j < d; j++)
			{
				int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
				hist1[idx] += hist1[idx + n];
				hist1[idx + 1] += hist1[idx + n + 1];
				hist2[idx] += hist2[idx + n];
				hist2[idx + 1] += hist2[idx + n + 1];
				hist3[idx] += hist3[idx + n];
				hist3[idx + 1] += hist3[idx + n + 1];
				for (k = 0; k < n; k++) {
					dst[(i*d + j)*n + k] = hist1[idx + k];
					dst[(i*d + j)*n + k + d * d * n] = hist2[idx + k];
					dst[(i*d + j)*n + k + d * d * n * 2] = hist3[idx + k];
				}
			}

		normalizeHistogram(dst, d, n);
	}

	void CHoNI::calculateIntensity(int band, float bins_per_intensity, int d, int n, float *hist, int radius, float cos_t, float sin_t, Point pt, int rows, int cols, float *CBin, float *RBin, const Mat& img, int histlen, float *intensity) const
	{

		int i, j, k;

		for (i = 0; i < histlen; i++)
		{
			hist[i] = 0;
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
					intensity[k] = img.at<Vec3f>(r, c)[band];

					ibar += intensity[k];
					ibar2 += intensity[k] * intensity[k];

					k++;
				}
			}

		int len = k;

		ibar = ibar / (float)len;
		ibar2 = ibar2 / (float)len;

		float isig = sqrt(ibar2 - ibar*ibar);

		float bias = 127.5f - ibar;
		float gain = 64.f / isig;

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
	}


	//////////////////////////////////////////////////////////////////////////////////////////

	CHoNI::CHoNI()
	{
	}

}
