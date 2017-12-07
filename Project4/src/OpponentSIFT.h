//-------------------------------------------------------------------------
// Name: OpponentSIFT.cpp
// Author: Clark Olson, Siqi Zhang
// Description: OpponentSIFT descriptor class, inherited from RGBSIFT
//	Opponenet SIFT generates descriptors by stacking SIFT descriptors for
//	the opponent color channels.
// Methods:
//			OpponentSIFT()
//			operator()
//			convertBGRImageToOpponentColorSpace()
//-------------------------------------------------------------------------
#ifndef __OPENCV_OpponentSIFT_H__
#define __OPENCV_OpponentSIFT_H__

#include "RGBSIFT.h"
using namespace std;
using namespace cv;
#ifdef __cplusplus

namespace cv
{

	class CV_EXPORTS_W OpponentSIFT : public RGBSIFT
	{
	public:
//------------------------------------create()-----------------------------------------
// create a pointer to the OpponentSIFT object
//Precondition: None
//Postcondition: the pointer is created
//-------------------------------------------------------------------------------------
		CV_WRAP static Ptr<OpponentSIFT> create()
		{
			return makePtr<OpponentSIFT>(OpponentSIFT());
		};

		CV_WRAP explicit OpponentSIFT();

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
		void operator()(InputArray _image, InputArray _mask,
			vector<KeyPoint>& keypoints,
			OutputArray _descriptors,
			bool useProvidedKeypoints) const;

//------------------------------ convertBGRImageToOpponentColorSpace ----------
// Convert the BGR image to opponent color space 
// Preconditions:  1. bgrImage must be valid
//				   2. opponentChannels is a valid refernce
// Postconditions: opponentChannels contains new image in opponent color space
//-----------------------------------------------------------------------------
		static void convertBGRImageToOpponentColorSpace(Mat& bgrImage);

	protected:
		virtual void normalizeHistogram(float *dst, int d, int n) const;
	};

} /* namespace cv */

#endif /* __cplusplus */
#endif
/* End of file. */