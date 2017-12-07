
///**********************************************************************************************\
//Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
//Below is the original copyright.
//
////    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
////    All rights reserved.
//
//\**********************************************************************************************/


#include "SPIN.h"
using namespace cv;
using namespace xfeatures2d;

// constructor
SPIN::SPIN()
{
	// do nothing
}

void SPIN::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const {
	Point pt(cvRound(ptf.x), cvRound(ptf.y));
	float cos_t = cosf(ori*(float)(CV_PI / 180));
	float sin_t = sinf(ori*(float)(CV_PI / 180));

	// the number of intensity bins (width of histogram)
	int intensity_bins = n;

	// the number of distance bins (width of histogram)
	int distance_bins = d;
	int grid_width = 4;

	// width of the whole histogram
	float hist_width = VanillaSIFT::SIFT_DESCR_SCL_FCTR * scl;

	int radius = int((grid_width + 1)*hist_width / sqrt(CV_PI));

	int circ_radius = (int) (grid_width*hist_width / sqrt(CV_PI));

	// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int)sqrt((double)img.cols*img.cols + img.rows*img.rows));

	// this is the number of intensity bins per unit
	float bins_per_intensity = (float)intensity_bins / 255.f;

	// this is the number of distance bins per pixel
	float bins_per_distance = (float)distance_bins / circ_radius;

	// the "soft widths" of the histograms
	float alpha = (float)circ_radius / (float)distance_bins;
	float beta = 255.f / (float)intensity_bins;

	int i;
	int j;
	int k;

	int len = (radius * 2 + 1)*(radius * 2 + 1);

	int histlen = intensity_bins*distance_bins;
	int rows = img.rows, cols = img.cols;

	AutoBuffer<float> buf(len * 2 + distance_bins);

	// the distance histogram
	float *dist = buf;

	// the intensity histogram
	float *intensity = dist + len;

	// array for tracking the sum of weights in each bin
	float *sums = intensity + len;

	for (int i = 0; i < distance_bins; i++) {
		sums[i] = 0;
	}

	for (int i = 0; i < distance_bins * intensity_bins; i++) {
		dst[i] = 0;
	}

	float ibar = 0, ibar2 = 0;

	for (i = -radius, k = 0; i <= radius; i++) {
		for (j = -radius; j <= radius; j++) {

			int r = pt.y + i, c = pt.x + j;

			float distance = (float)sqrt(i*i + j*j);

			if (r > -1 && c > -1 && r < rows - 1 && c < cols - 1 && distance <= radius) {
				dist[k] = distance;

				// setting the intensity value of each pixel 
				intensity[k] = img.at<VanillaSIFT::sift_wt>(r, c);

				ibar += intensity[k];
				ibar2 += intensity[k] * intensity[k];

				k++;
			}
		}
	}

	len = k;

	ibar = ibar / (float)len;
	ibar2 = ibar2 / (float)len;

	float isig = sqrt(ibar2 - ibar*ibar);

	float bias = 127.5f - ibar;
	float gain = 64.f / isig;

	for (k = 0; k < len; k++) {
		// Normalize intensity
		intensity[k] = (intensity[k] - ibar)*gain + ibar + bias;

		int distance_bin = (int)(dist[k] * bins_per_distance);
		// this handles edge case when distance_bin is at the radius length
		distance_bin = min(distance_bins - 1, distance_bin);

		int intensity_bin = (int)(intensity[k] * bins_per_intensity);
		// this handles edge case when intensity_bin is at the radius length
		intensity_bin = min(intensity_bins - 1, intensity_bin);

		int imin = max(0, distance_bin - 1);
		// subtracting 1 because distance_bins is bin index and we need bin location
		int imax = min(distance_bins - 1, distance_bin + 1);

		int jmin = max(0, intensity_bin - 1);
		// subtracting 1 because intensity_bins is bin index and we need bin location
		int jmax = min(intensity_bins - 1, intensity_bin + 1);


		for (i = imin; i <= imax; i++) {
			for (j = jmin; j <= jmax; j++) {

				// index in destination (dst) histogram
				int index = i * intensity_bins + j;

				// calculate the denominators
				float distanceDenominator = 2 * alpha * alpha;
				float intensityDenominator = 2 * beta * beta;

				// calculate distance from keypoint
				float i_dist = (i + 0.5f) * alpha;
				float j_dist = (j + 0.5f) * beta;

				// calculate numerators
				float distanceNumerator = pow(dist[k] - i_dist, 2);
				float intensityNumerator = pow(intensity[k] - j_dist, 2);

				float distanceCalculation = distanceNumerator / distanceDenominator;
				float intensityCalculation = intensityNumerator / intensityDenominator;

				// calculate the exponential
				float expo = exp(-distanceCalculation - intensityCalculation);

				// add final calculation to sums array for normalization of weights
				sums[i] += expo;

				// add final calculation to the destination array
				dst[index] += expo;

			}
		}
	}

	for (int i = 0; i < distance_bins; i++) {
		for (int j = 0; j < intensity_bins; j++) {
			dst[i * intensity_bins + j] /= std::max(sums[i], FLT_EPSILON);
		}
	}


	// copy histogram to the descriptor,
	// apply hysteresis thresholding
	// and scale the result, so that it can be easily converted
	// to byte array
	float nrm2 = 0;
	len = distance_bins * intensity_bins;
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
		dst[k] = dst[k] * nrm2;
	}
}

//------------------------------------calcDescriptors()--------------------------------
// set up variables and call calcSIFTDescriptor() to compute descriptors
//Precondition: the following parameters must be correclty defined.
//parameters:
//gpyr: gaussian pyramid
//keypoints: computed keypoints
//descriptors: empty descriptors vector
//nOctaveLayers: number of octave layers
//firstOctave: index of first octave
//Postcondition: descriptors are assigned
//-------------------------------------------------------------------------------------
void SPIN::calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
	Mat& descriptors, int nOctaveLayers, int firstOctave) const
{
	int d = NUM_DISTANCE_BINS, n = NUM_INTENSITY_BINS;

	for (size_t i = 0; i < keypoints.size(); i++)
	{
		KeyPoint kpt = keypoints[i];
		int octave, layer;
		float scale;
		unpackOctave(kpt, octave, layer, scale);
		CV_Assert(octave >= firstOctave && layer <= nOctaveLayers + 2);
		float size = kpt.size*scale;
		Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
		const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

		float angle = 360.f - kpt.angle;
		if (std::abs(angle - 360.f) < FLT_EPSILON)
			angle = 0.f;

		//printf("octave: %3d     scale: %5.1f     size: %5.1f\n", octave, scale, size*0.5f);
		calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
	}
}

//------------------------------------descriptorSize()---------------------------------
// ! returns the descriptor size in floats
//Precondition: None
//Postcondition: the descriptor size is returned in floats
//-------------------------------------------------------------------------------------
int SPIN::descriptorSize() const
{
	return NUM_DISTANCE_BINS*NUM_INTENSITY_BINS;
}