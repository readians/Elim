#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <sstream>
#include <string.h>
#include <math.h>

using namespace cv;
using namespace std;



int main( int argc, char** argv )
{
	/*-------------------------------------------------------------------------------------*/
	VideoCapture cap; // open the video camera for reading (use a string for a file)
	Mat frame, gray_frame;
	Mat prev,next;
	Mat bgr;//CV_32FC3 matrix
	Mat flow;
	Mat xy[2];
	Mat magnitude, angle;
	Mat _hsv[3], hsv;
	Mat I[15],SM;
	namedWindow("Coherency Based Spatio-Temporal SM (Up to 15 frames)", CV_WINDOW_AUTOSIZE);
	namedWindow("Optical Flow",1);
	vector<float> descriptorsValues;
	vector<Point> locations;
	HOGDescriptor d(Size(640,480),Size(16,16),Size(16,16),Size(16,16),9,0,-1,HOGDescriptor::L2Hys, 0.2, false, HOGDescriptor::DEFAULT_NLEVELS);
				// Size(640,480), //winSize
				// Size(16,16), //blocksize
				// Size(16,16), //blockStride, if equal to blocksize there is no overlap
				// Size(16,16), //cellSize
				// 9, //nbins,
				// 0, //derivAper,
				// -1, //winSigma,
				// 0, //histogramNormalizationType,
				// 0.2, //L2HysThresh,
				// false //gamma correction,
				// nlevels=64
	int i,j,k,N=0;	
	int WIDTH,HEIGHT;
	int patch_x;
	int patch_y;
	int maxLevel=3;
	int flags = 0;
	float ***p;
	float **C;
	float s;
	double mag_max;

	cap.open(0);
	
	WIDTH = 640; //640 on hp-pavillion webcam
	HEIGHT = 480; //480 on hp-pavillion webcam
	patch_x = WIDTH/16; //40 on hp-pavillion webcam
	patch_y = HEIGHT/16; //30 on hp-pavillion webcam

	/*C = new float*[patch_y];
	for(i = 0; i < patch_y; ++i)
		C[i] = new float[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			C[i][j] = 0;
	p = new float**[patch_y];
	for(i = 0; i < patch_y; i++)
	{
		p[i] = new float*[patch_x];
		for(j = 0; j < patch_x; j++)
		{
			p[i][j] = new float[9];
		}
	}
	for(i = 0; i < patch_y; i++)
		for(j = 0; j < patch_x; j++)
			for(k=0;k<9;k++)
				p[i][j][k] = 0;*/

	while(1)
    {
		//for(N=0;N<15;N++)
		//{
			cap >> frame; // read a new frame from video
			
			cvtColor(frame, gray_frame, CV_BGR2GRAY);
		
			gray_frame.convertTo(next, -1, 1.2, 0);
			
			if(N > 0)
			{
				calcOpticalFlowFarneback(prev, next, flow, 0.5, 7, 15, 3, 5, 1.2, 0);

				split(flow, xy);

				//calculate angle and magnitude
				cartToPolar(xy[0], xy[1], magnitude, angle, true);

				//translate magnitude to range [0;1]
				minMaxLoc(magnitude, 0, &mag_max);
				magnitude.convertTo(magnitude, -1, 1.0/mag_max);

				//build hsv image
				_hsv[0] = angle;
				_hsv[1] = Mat::ones(angle.size(), CV_32F);
				_hsv[2] = magnitude;
				merge(_hsv, 3, hsv);

				//convert to BGR and show
				cvtColor(hsv, bgr, COLOR_HSV2BGR);
				imshow("Optical Flow",bgr);
			}

			swap(prev, next);
			
			N = 1;

			/*if(N == 14)
			{
				d.compute(I[N], descriptorsValues, Size(0,0), Size(0,0), locations);
		
				for(i=0;i<patch_x;i++)
				{
					for(j=0;j<patch_y;j++)
					{
						s = 0;
						for(k=0;k<9;k++)
							s += descriptorsValues[(i * patch_y + j) * 9 + k];
						for(k=0;k<9;k++)
						{
							p[j][i][k] = descriptorsValues[(i * patch_y + j) * 9 + k]/s;
							C[j][i] += p[j][i][k]*log10(p[j][i][k]);
						}
						C[j][i] = -C[j][i];
					}
				}
			}*/
		//}

		//SM = formula della mappa SM;

		//imshow("Coherency Based Spatio-Temporal SM (Up to 15 frames)",SM);

		/*for(i=0;i<patch_y;i++)
				for(j=0;j<patch_x;j++)
					C[i][j] = 0;*/
		
		if(waitKey(10) == 27) //wait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
		{
			system("PAUSE");
			cout << "esc key is pressed by user" << endl; 
            break; 
		}
    }
}
