
/**********************************************************************************************\
Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/SIFT/
Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

#include "OpponentSIFT.h"

namespace cv
{
	
	// constructor
	OpponentSIFT::OpponentSIFT()
	{
		// do nothing
	}

//------------------------------------operator()---------------------------------------
// Overloading operator() to run the algorithm using color image:
// 1. compute keypoints using local extrema of Dog space
// 2. compute descriptors with keypoints and color pyramid
//Precondition: the following parameters must be correclty defined.
//parameters:
	//_image: color image base
	//_mask: image mask
	//keypoints: keypoints of the image
	//_descriptors: descriptors
	//useProvidedKeypoints: bool indicating whether using provided keypoints
//Postcondition: 1. keypoints are assigned
//				 2. descriptors are calculated and assigned
//-------------------------------------------------------------------------------------
	void OpponentSIFT::operator()(InputArray _image, InputArray _mask,
		vector<KeyPoint>& keypoints,
		OutputArray _descriptors,
		bool useProvidedKeypoints) const
	{
		int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
		Mat image = _image.getMat(), mask = _mask.getMat();

		if (image.empty() || image.depth() != CV_8U)
			CV_Error(CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

		if (!mask.empty() && mask.type() != CV_8UC1)
			CV_Error(CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)");

		if (useProvidedKeypoints)
		{
			firstOctave = 0;
			int maxOctave = INT_MIN;
			for (size_t i = 0; i < keypoints.size(); i++)
			{
				int octave, layer;
				float scale;
				unpackOctave(keypoints[i], octave, layer, scale);
				firstOctave = std::min(firstOctave, octave);
				maxOctave = std::max(maxOctave, octave);
				actualNLayers = std::max(actualNLayers, layer - 2);
			}

			firstOctave = std::min(firstOctave, 0);
			CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
			actualNOctaves = maxOctave - firstOctave + 1;
		}
		// base is a grey image
		Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
		//initialize color image
		Mat colorBase = createInitialColorImage(image, firstOctave < 0, (float)sigma);
		convertBGRImageToOpponentColorSpace(colorBase);

		vector<Mat> gpyr, dogpyr, colorGpyr; // colorGpyr is a gaussian pyramid for color image
		int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) / log(2.) - 2) - firstOctave;

		//double t, tf = getTickFrequency();
		//t = (double)getTickCount();
		buildGaussianPyramid(base, gpyr, nOctaves);
		buildDoGPyramid(gpyr, dogpyr);
		// build color gaussian pyramid
		buildGaussianPyramid(colorBase, colorGpyr, nOctaves);
		//t = (double)getTickCount() - t;
		//printf("pyramid construction time: %g\n", t*1000./tf);

		if (!useProvidedKeypoints)
		{
			//t = (double)getTickCount();
			findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
			KeyPointsFilter::removeDuplicated(keypoints);

			if (nfeatures > 0)
				KeyPointsFilter::retainBest(keypoints, nfeatures);
			//t = (double)getTickCount() - t;
			//printf("keypoint detection time: %g\n", t*1000./tf);

			if (firstOctave < 0)
				for (size_t i = 0; i < keypoints.size(); i++)
				{
				KeyPoint& kpt = keypoints[i];
				float scale = 1.f / (float)(1 << -firstOctave);
				kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
				kpt.pt *= scale;
				kpt.size *= scale;
				}

			if (!mask.empty())
				KeyPointsFilter::runByPixelsMask(keypoints, mask);
		}
		else
		{
			// filter keypoints by mask
			KeyPointsFilter::runByPixelsMask( keypoints, mask );
		}

		if (_descriptors.needed())
		{
			//t = (double)getTickCount();
			int dsize = descriptorSize();
			_descriptors.create((int)keypoints.size(), dsize, CV_32F);
			Mat descriptors = _descriptors.getMat();
			calcDescriptors(colorGpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
			//t = (double)getTickCount() - t;
			//printf("descriptor extraction time: %g\n", t*1000./tf);
		}
	}


//------------------------------ convertBGRImageToOpponentColorSpace ----------
// Convert the BGR image to opponent color space 
// Preconditions:  1. bgrImage must be valid
//				   2. opponentChannels is a valid refernce
// Postconditions: opponentChannels contains new image in opponent color space
//-----------------------------------------------------------------------------
	void OpponentSIFT::convertBGRImageToOpponentColorSpace(Mat& bgrImage)
	{
		if (bgrImage.type() != CV_32FC3)
			CV_Error(CV_StsBadArg, "input image must be an BGR image of type CV_32FC3");
		
		for (int y = 0; y < bgrImage.rows; ++y) {
			for (int x = 0; x < bgrImage.cols; ++x) {
				Vec3f v = bgrImage.at<Vec3f>(y, x);
				float& b = v[0];
				float& g = v[1];
				float& r = v[2];

				bgrImage.at<Vec3f>(y, x)[0] = ((r - g) / sqrtf(2));	
				bgrImage.at<Vec3f>(y, x)[1] = ((r + g - 2 * b) / sqrtf(6));
				bgrImage.at<Vec3f>(y, x)[2] = ((r + g + b) / sqrtf(3));
			}
		}
	}
}

void OpponentSIFT::normalizeHistogram(float *dst, int d, int n) const {
	float nrm1sqr = 0, nrm2sqr = 0, nrm3sqr = 0;
	int len = d*d*n;
	for (int k = 0; k < len; k++) {
		nrm1sqr += dst[k] * dst[k];
		nrm2sqr += dst[k + len] * dst[k + len];
		nrm3sqr += dst[k + 2 * len] * dst[k + 2 * len];
	}
	float nrm2 = max(nrm1sqr, max(nrm2sqr, nrm3sqr));
	float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;

	nrm2 = 0;
	for (int i = 0; i < 3 * len; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val * val;
	}
	nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

	for (int k = 0; k < 3 * len; k++)
	{
		dst[k] = dst[k] * nrm2;
	}
}