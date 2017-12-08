//See other files for SIFT implementation credits

// PSIFT.CPP
// Alex Lee, Fengjuan Qiu, John Zoeller
//
// Uses the RGB SIFT implementation.  Descriptor calculation modified to 
// "pool" the color bands by summing them.

#include "PSIFT.h"

namespace cv
{
	//-------------------------------------------------------------------------------------
	//img: color image
	//ptf: keypoint
	//ori: angle(degree) of the keypoint relative to the coordinates, clockwise
	//scl: radius of meaningful neighborhood around the keypoint 
	//d: newsift descr_width, 4 in this case
	//n: SIFT_descr_hist_bins, 8 in this case
	//dst: descriptor array to pass in
	//changes: 1. img now is a color image
	void PSIFT::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl,
		int d, int n, float* dst) const
	{
		Point pt(cvRound(ptf.x), cvRound(ptf.y));
		float cos_t = cosf(ori*(float)(CV_PI / 180));
		float sin_t = sinf(ori*(float)(CV_PI / 180));
		float bins_per_rad = n / 360.f;
		float exp_scale = -1.f / (d * d * 0.5f);
		float hist_width = SIFT_DESCR_SCL_FCTR * scl;
		int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
		// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
		radius = std::min(radius, (int)sqrt((double)img.cols*img.cols + img.rows*img.rows));
		cos_t /= hist_width;
		sin_t /= hist_width;

		int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1), histlen = (d + 2)*(d + 2)*(n + 2);
		int rows = img.rows, cols = img.cols;

		AutoBuffer<float> buf(len * 12 + histlen * 3);
		float *X1 = buf, *Y1 = X1 + len, *X2 = Y1 + len, *Y2 = X2 + len, *X3 = Y2 + len, *Y3 = X3 + len;
		float *Mag1 = Y1, *Mag2 = Y2, *Mag3 = Y3, *Ori1 = Mag3 + len, *Ori2 = Ori1 + len, *Ori3 = Ori2 + len, *W = Ori3 + len;
		float *RBin = W + len, *CBin = RBin + len, *hist1 = CBin + len, *hist2 = hist1 + histlen, *hist3 = hist2 + histlen;

		float *pooledHist = new float[128];

		for (i = 0; i < d + 2; i++)
		{
			for (j = 0; j < d + 2; j++) {
				for (k = 0; k < n + 2; k++) {
					hist1[(i*(d + 2) + j)*(n + 2) + k] = 0.;
					hist2[(i*(d + 2) + j)*(n + 2) + k] = 0.;
					hist3[(i*(d + 2) + j)*(n + 2) + k] = 0.;
				}
			}
		}

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
					X1[k] = (img.at<Vec3f>(r, c + 1)[0] - img.at<Vec3f>(r, c - 1)[0]);
					Y1[k] = (img.at<Vec3f>(r - 1, c)[0] - img.at<Vec3f>(r + 1, c)[0]);
					X2[k] = (img.at<Vec3f>(r, c + 1)[1] - img.at<Vec3f>(r, c - 1)[1]);
					Y2[k] = (img.at<Vec3f>(r - 1, c)[1] - img.at<Vec3f>(r + 1, c)[1]);
					X3[k] = (img.at<Vec3f>(r, c + 1)[2] - img.at<Vec3f>(r, c - 1)[2]);
					Y3[k] = (img.at<Vec3f>(r - 1, c)[2] - img.at<Vec3f>(r + 1, c)[2]);
					RBin[k] = rbin; CBin[k] = cbin;
					W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
					k++;
				}
			}

		len = k;
		hal::fastAtan2(Y1, X1, Ori1, len, true);
		hal::magnitude(X1, Y1, Mag1, len);
		hal::fastAtan2(Y2, X2, Ori2, len, true);
		hal::magnitude(X2, Y2, Mag2, len);
		hal::fastAtan2(Y3, X3, Ori3, len, true);
		hal::magnitude(X3, Y3, Mag3, len);
		hal::exp(W, W, len);

		for (k = 0; k < len; k++)
		{
			float rbin = RBin[k], cbin = CBin[k];
			float obin1 = (Ori1[k] - ori)*bins_per_rad;
			float mag1 = Mag1[k] * W[k];
			float obin2 = (Ori2[k] - ori)*bins_per_rad;
			float mag2 = Mag2[k] * W[k];
			float obin3 = (Ori3[k] - ori)*bins_per_rad;
			float mag3 = Mag3[k] * W[k];

			int r0 = cvFloor(rbin);
			int c0 = cvFloor(cbin);
			int o01 = cvFloor(obin1);
			int o02 = cvFloor(obin2);
			int o03 = cvFloor(obin3);
			rbin -= r0;
			cbin -= c0;
			obin1 -= o01;
			obin2 -= o02;
			obin3 -= o03;

			if (o01 < 0)
				o01 += n;
			if (o01 >= n)
				o01 -= n;
			if (o02 < 0)
				o02 += n;
			if (o02 >= n)
				o02 -= n;
			if (o03 < 0)
				o03 += n;
			if (o03 >= n)
				o03 -= n;

			// histogram update using tri-linear interpolation
			float v1_r1 = mag1*rbin, v1_r0 = mag1 - v1_r1;
			float v1_rc11 = v1_r1*cbin, v1_rc10 = v1_r1 - v1_rc11;
			float v1_rc01 = v1_r0*cbin, v1_rc00 = v1_r0 - v1_rc01;
			float v1_rco111 = v1_rc11*obin1, v1_rco110 = v1_rc11 - v1_rco111;
			float v1_rco101 = v1_rc10*obin1, v1_rco100 = v1_rc10 - v1_rco101;
			float v1_rco011 = v1_rc01*obin1, v1_rco010 = v1_rc01 - v1_rco011;
			float v1_rco001 = v1_rc00*obin1, v1_rco000 = v1_rc00 - v1_rco001;

			float v2_r1 = mag2*rbin, v2_r0 = mag2 - v2_r1;
			float v2_rc11 = v2_r1*cbin, v2_rc10 = v2_r1 - v2_rc11;
			float v2_rc01 = v2_r0*cbin, v2_rc00 = v2_r0 - v2_rc01;
			float v2_rco111 = v2_rc11*obin2, v2_rco110 = v2_rc11 - v2_rco111;
			float v2_rco101 = v2_rc10*obin2, v2_rco100 = v2_rc10 - v2_rco101;
			float v2_rco011 = v2_rc01*obin2, v2_rco010 = v2_rc01 - v2_rco011;
			float v2_rco001 = v2_rc00*obin2, v2_rco000 = v2_rc00 - v2_rco001;

			float v3_r1 = mag3*rbin, v3_r0 = mag3 - v3_r1;
			float v3_rc11 = v3_r1*cbin, v3_rc10 = v3_r1 - v3_rc11;
			float v3_rc01 = v3_r0*cbin, v3_rc00 = v3_r0 - v3_rc01;
			float v3_rco111 = v3_rc11*obin3, v3_rco110 = v3_rc11 - v3_rco111;
			float v3_rco101 = v3_rc10*obin3, v3_rco100 = v3_rc10 - v3_rco101;
			float v3_rco011 = v3_rc01*obin3, v3_rco010 = v3_rc01 - v3_rco011;
			float v3_rco001 = v3_rc00*obin3, v3_rco000 = v3_rc00 - v3_rco001;

			int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o01;
			hist1[idx] += v1_rco000;
			hist1[idx + 1] += v1_rco001;
			hist1[idx + (n + 2)] += v1_rco010;
			hist1[idx + (n + 3)] += v1_rco011;
			hist1[idx + (d + 2)*(n + 2)] += v1_rco100;
			hist1[idx + (d + 2)*(n + 2) + 1] += v1_rco101;
			hist1[idx + (d + 3)*(n + 2)] += v1_rco110;
			hist1[idx + (d + 3)*(n + 2) + 1] += v1_rco111;
			idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o02;
			hist2[idx] += v2_rco000;
			hist2[idx + 1] += v2_rco001;
			hist2[idx + (n + 2)] += v2_rco010;
			hist2[idx + (n + 3)] += v2_rco011;
			hist2[idx + (d + 2)*(n + 2)] += v2_rco100;
			hist2[idx + (d + 2)*(n + 2) + 1] += v2_rco101;
			hist2[idx + (d + 3)*(n + 2)] += v2_rco110;
			hist2[idx + (d + 3)*(n + 2) + 1] += v2_rco111;
			idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o03;
			hist3[idx] += v3_rco000;
			hist3[idx + 1] += v3_rco001;
			hist3[idx + (n + 2)] += v3_rco010;
			hist3[idx + (n + 3)] += v3_rco011;
			hist3[idx + (d + 2)*(n + 2)] += v3_rco100;
			hist3[idx + (d + 2)*(n + 2) + 1] += v3_rco101;
			hist3[idx + (d + 3)*(n + 2)] += v3_rco110;
			hist3[idx + (d + 3)*(n + 2) + 1] += v3_rco111;
		}

		// finalize histogram, since the orientation histograms are circular
		//d = 4
		//n = 8
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

					pooledHist[(i*d + j)*n + k] = hist1[idx + k] + hist2[idx + k] + hist3[idx + k];
				}
			}
		// copy histogram to the descriptor,
		// apply hysteresis thresholding
		// and scale the result, so that it can be easily converted
		// to byte array

		normalizePoolHist(pooledHist, d, n);
		dst = new float[128];
		dst = pooledHist; 
	}


	// Purpose	:	Normalize a histogram 
	// Preconx	:	All params have values
	// Postconx	:	Histogram properly normalized
	// Params	:	
		// float * pooled	:	vector of 128 dimensions
		// int d			:	
		// int n			: 
	// Return	:	void
	void PSIFT::normalizePoolHist(float *pooled, int d, int n) const {
		float nrm1sqr = 0;
		int len = d*d*n;			//128

		for (int k = 0; k < len; k++)
			nrm1sqr += pooled[k] * pooled[k];


		float thr1 = std::sqrt(nrm1sqr)*SIFT_DESCR_MAG_THR;
		nrm1sqr = 0;

		for (int k = 0; k < len; k++) {
			float val = std::min(pooled[k], thr1);
			pooled[k] = val;
			nrm1sqr += val*val;
		}

		// Factor of three added below to make vector lengths comparable with SIFT
		nrm1sqr = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(3 * nrm1sqr), FLT_EPSILON);

		for (int k = 0; k < len; k++)
			pooled[k] = (pooled[k] * nrm1sqr);
	}

	PSIFT::PSIFT() {
}                
}