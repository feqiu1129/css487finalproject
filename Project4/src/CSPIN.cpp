#include "CSPIN.h"


CSPIN::CSPIN()
{

}

void CSPIN::normalizeHistogram(float *dst, int d, int n) const {
	float nrm1sqr = 0, nrm2sqr = 0, nrm3sqr = 0;
	int len = d*n;
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


void CSPIN::calcSIFTDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst) const {

	Point pt(cvRound(ptf.x), cvRound(ptf.y));

	int i, k;
	
	int NUM_BANDS = 3;

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
	float bins_per_distance = (float) distance_bins / circ_radius;

	// the "soft widths" of the histograms
	float alpha = (float)circ_radius / (float)distance_bins;
	float beta = 255.f / (float)intensity_bins;

	int len = (radius * 2 + 1)*(radius * 2 + 1);

	int histlen = intensity_bins*distance_bins;
	int rows = img.rows, cols = img.cols;
		
	// adding 1 to NUM_BANDS to make space for each histogram
	AutoBuffer<float> buf(len * (NUM_BANDS + 1) + distance_bins);

	// the distance histogram
	float *dist = buf;

	// the intensity histograms
	float *red = dist + len;
	float *green = red + len;
	float *blue = green + len;

	for (int i = 0; i < distance_bins * intensity_bins * NUM_BANDS; i++) {
		dst[i] = 0;
	}

	calcBand(img, dst, 0, blue, dist, pt, radius, len, intensity_bins, distance_bins, bins_per_intensity, bins_per_distance, alpha, beta);
	calcBand(img, dst, 1, green, dist, pt, radius, len, intensity_bins, distance_bins, bins_per_intensity, bins_per_distance, alpha, beta);
	calcBand(img, dst, 2, red, dist, pt, radius, len, intensity_bins, distance_bins, bins_per_intensity, bins_per_distance, alpha, beta);

	const bool separate = true; // should be true, I think

	if (separate) {
		normalizeHistogram(dst, d, n);
	}
	else {

		// copy histogram to the descriptor,
		// apply hysteresis thresholding
		// and scale the result, so that it can be easily converted
		// to byte array
		float nrm2 = 0;
		len = distance_bins * intensity_bins * NUM_BANDS;
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
			dst[k] = (dst[k] * nrm2);
		}
	}
}


void CSPIN::calcBand(const Mat& img, 
							 float *dst, 
							 int band, 
							 float *intensity, 
							 float *dist, 
							 Point pt, 
							 int radius,
							 int len, 
							 int intensity_bins,
							 int distance_bins, 
							 float bins_per_intensity,
							 float bins_per_distance, 
							 float alpha,
							 float beta) const {

	int i;
	int j;
	int k;


	AutoBuffer<float> sums(distance_bins);

	// array for tracking the sum of weights in each bin

	for(int i = 0; i < distance_bins; i++) {
		sums[i] = 0;
	}
	
	float ibar = 0, ibar2 = 0;

	int rows = img.rows, cols = img.cols;

	for (i = -radius, k = 0; i <= radius; i++) {
		for (j = -radius; j <= radius; j++) {

			int r = pt.y + i, c = pt.x + j;

			float distance = (float)sqrt(i*i + j*j);

			if (r > -1 && c > -1 && r < rows - 1 && c < cols - 1 && distance <= radius) {
				dist[k] = distance;

				// setting the intensity value of each pixel 
				intensity[k] = img.at<Vec3f>(r, c)[band];

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
				dst[(band * 128)+index] += expo;

			}
		}
	}

	for(int i = 0; i < distance_bins; i++) {
		for(int j = 0; j < intensity_bins; j++) {
			dst[(band * 128) + (i * intensity_bins + j)] /= std::max(sums[i], FLT_EPSILON);
		}
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
void CSPIN::calcDescriptors(const std::vector<Mat>& gpyr, const std::vector<KeyPoint>& keypoints,
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
int CSPIN::descriptorSize() const
{
	return 3*NUM_DISTANCE_BINS*NUM_INTENSITY_BINS;
}