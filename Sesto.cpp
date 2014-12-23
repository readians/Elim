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

float costFunMAD(Mat curr, Mat ref, int n){
	int i,j;
	float err = 0;

	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			err = err + abs(curr.data[curr.cols*i + j] - ref.data[ref.cols*i + j]);

	return err / (n*n);
}

float min_arr(float *a){
	float min = 65537;

	for(int i=0;i<sizeof(a)/sizeof(int);i++)
		if (a[i] < min)
			min = a[i];

	return min;
}

int min_arr_point(float *a){
	int point = 0;
	float min = a[0];

	for(int i=0;i<sizeof(a)/sizeof(int);i++)
		if (a[i] < min)
			point = i;
	
	return point;
}

void BMA(Mat A, Mat B, int blocksize, int p, int **MVx, int **MVy)
{
	//Mat flow = Mat::zeros(A.rows/blocksize,A.cols/blocksize, A.type());
	Mat c,r;
	int WIDTH = A.cols;
	int HEIGHT = A.rows;
	int patch_x = WIDTH/blocksize;
	int patch_y = HEIGHT/blocksize;
	int i,j,k,l;
	float MAD, minMAD = (float)blocksize;

	/*Start computing motion vectors for each block of the frame*/
	for(i=0;i<=HEIGHT-blocksize;i+=blocksize)
	{
		for(j=0;j<=WIDTH-blocksize;j+=blocksize)
		{
			patch_x = j/blocksize;
			patch_y = i/blocksize;
			Mat startB(A, Rect(j,i,blocksize,blocksize));
			startB.copyTo(c);
			for(k=i-p;k<=i+p;k++)
			{
				for(l=j-p;l<=j+p;l++)
				{
					if(l>=0 && l+blocksize<=WIDTH && k>=0 && k+blocksize<=HEIGHT)
					{
						Mat searchB(B, Rect(l,k,blocksize,blocksize));
						searchB.copyTo(r);
						MAD = costFunMAD(c,r,blocksize);
						if((MAD<minMAD) || (l==j && k==i && MAD==minMAD))
						{
							minMAD = MAD;
							MVx[patch_y][patch_x] = (l-j);
							MVy[patch_y][patch_x] = (k-i);
						}
						searchB.release();
					}
				}
			}
			startB.release();
			minMAD = (float)blocksize;
		}
	}
}


int main( int argc, char** argv )
{
	VideoCapture cap; // open the video camera for reading (use a string for a file)
	Mat frame, gray_frame;
	Mat prev,next;
	Mat flow;
	Mat SM;
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
	int **MVx,**MVy;
	float ***p;
	float **C;
	float s;

	//namedWindow("Coherency Based STSM (Up to 15 frames)", CV_WINDOW_AUTOSIZE);
	namedWindow("Optical Flow",1);
	
	cap.open(0);
	
	WIDTH = 640; //640 on hp-pavillion webcam
	HEIGHT = 480; //480 on hp-pavillion webcam
	patch_x = WIDTH/16; //40 on hp-pavillion webcam
	patch_y = HEIGHT/16; //30 on hp-pavillion webcam
	
	C = new float*[patch_y];
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
				p[i][j][k] = 0;
	MVx = new int*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MVx[i] = new int[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			MVx[i][j] = 0;
	MVy = new int*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MVy[i] = new int[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			MVy[i][j] = 0;

	/*************************************************************************/
	prev = Mat::zeros(480,640, CV_8UC1);
	next = Mat::zeros(480,640, CV_8UC1);
	for(i=0;i<=15;i++)
		for(j=0;j<=15;j++)
		{
			prev.data[WIDTH*i + j] = 255;
			next.data[WIDTH*(i+4) + (j+4)] = 255;
		}
	BMA(prev, next, 16, 7, MVx, MVy);
	for(i = 0; i < patch_y; i++)
		for(j = 0; j < patch_x; j++)
			if(MVx[i][j]!=0 || MVy[i][j]!=0)
				cout <<"("<<MVy[i][j]<<","<<MVx[i][j]<<") :"<<i<<","<<j<<endl;
	system("PAUSE");
	/**************************************************************************/
	while(1)
    	{
		//for(N=0;N<15;N++)
		//{
			cap >> frame; // read a new frame from video			
			
			cvtColor(frame, gray_frame, CV_BGR2GRAY);

			gray_frame.convertTo(next, -1, 1.2, 0);
			
			if(N > 0)
			{
				BMA(prev, next, 16, 7, MVx, MVy);
				imshow("Optical Flow", frame);
			}
			N = 1;
			swap(prev, next);
			
			if(N == 14)
			{
				d.compute(next, descriptorsValues, Size(0,0), Size(0,0), locations);
		
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
			}
		//}
		
		//SM = formula della mappa SM;

		//imshow("Coherency Based STSM (Up to 15 frames)",SM);

		for(i=0;i<patch_y;i++)
				for(j=0;j<patch_x;j++)
					C[i][j] = 0;
		
		if(waitKey(10) == 27) //wait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
		{
			system("PAUSE");
			cout << "esc key is pressed by user" << endl; 
            		break; 
		}
    	}
}
