#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/video/background_segm.hpp"
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <sstream>
#include <string.h>
#include <float.h>
#include <math.h>

using namespace cv;
using namespace std;

#define PI 3.14159265359

float costFunMAD(Mat curr, Mat ref, int n){
	int i,j;
	float err = 0;

	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			err = err + abs(curr.data[curr.cols*i + j] - ref.data[ref.cols*i + j]);

	return err / (n*n);
}

void ARPS(Mat A, Mat B, int blocksize, int p, int **MVx, int **MVy, float thres)
{
	//Mat flow = Mat::zeros(A.rows/blocksize,A.cols/blocksize, A.type());
	Mat C,R;
	int WIDTH = A.cols;
	int HEIGHT = A.rows;
	int patch_x = WIDTH/blocksize;
	int patch_y = HEIGHT/blocksize;
	int i,j,k,l,x,y,T,ind,maxind;
	int POFS[9][2]; //Points of first search
	int POLS[2]; //Point Of Local (second step) Search: 0 = y, 1 = x
	int ninth[2];
	float MAD, minMAD = 256;

	/*Start computing motion vectors for each block of the frame*/
	for(i=0;i<=HEIGHT-blocksize;i+=blocksize)
	{
		for(j=0;j<=WIDTH-blocksize;j+=blocksize)
		{
			patch_x = j/blocksize;
			patch_y = i/blocksize;
			Mat startB(A, Rect(j,i,blocksize,blocksize));
			startB.copyTo(C);
			
			/*ZMP*/
			Mat searchB(B, Rect(j,i,blocksize,blocksize));
			searchB.copyTo(R);
			MAD = costFunMAD(C,R,blocksize);
			searchB.release();
			if(MAD<thres)
			{
				MVx[patch_y][patch_x] = 0;
				MVy[patch_y][patch_x] = 0;
			}
			else
			{
				if(j==0)
				{
					T = 3;
					maxind = 8;
				}
				else
				{
					T = max(abs(MVx[patch_y][patch_x-1]),abs(MVy[patch_y][patch_x-1]));
					ninth[0] = MVy[patch_y][patch_x-1]; ninth[1] = MVx[patch_y][patch_x-1];
					maxind = 9;
				}
				POLS[0] = 0;
				POLS[1] = 0;
				POFS[0][0] = i;			POFS[0][1] = j-T;
				POFS[1][0] = i;			POFS[1][1] = j+T;
				POFS[2][0] = i-T;		POFS[2][1] = j;
				POFS[3][0] = i+T;		POFS[3][1] = j;
				POFS[4][0] = i-T;		POFS[4][1] = j-T;
				POFS[5][0] = i-T;		POFS[5][1] = j+T;
				POFS[6][0] = i+T;		POFS[6][1] = j-T;
				POFS[7][0] = i+T;		POFS[7][1] = j+T;
				POFS[8][0] = i+ninth[0];POFS[8][1] = j+ninth[1];

				/*Initial Search*/
				for(ind=0;ind<maxind;ind++)
				{
					k = POFS[ind][0];
					l = POFS[ind][1];
					if(l>=0 && l+blocksize<=WIDTH && k>=0 && k+blocksize<=HEIGHT)
					{
						Mat searchB(B, Rect(l,k,blocksize,blocksize));
						searchB.copyTo(R);
						MAD = costFunMAD(C,R,blocksize);
						if(MAD < minMAD)
						{
							MVy[patch_y][patch_x] = k-i;
							MVx[patch_y][patch_x] = l-j;
							minMAD = MAD;
							if(minMAD < thres)
								break;
						}
						searchB.release();
					}
				}

				POLS[0] = MVy[patch_y][patch_x];
				POLS[1] = MVx[patch_y][patch_x];

				/*Local search using pxp mask*/
				for(y=i+POLS[0]-p;y<=i+POLS[0]+p;y++)
				{
					if(minMAD==0)
						break;
					for(x=j+POLS[1]-p;x<=j+POLS[1]+p;x++)
					{
						if(x>=0 && x+blocksize<=WIDTH && y>=0 && y+blocksize<=HEIGHT)
						{
							Mat searchB(B, Rect(x,y,blocksize,blocksize));
							searchB.copyTo(R);
							MAD = costFunMAD(C,R,blocksize);
							if(MAD < minMAD)
							{
								MVy[patch_y][patch_x] = y-i;
								MVx[patch_y][patch_x] = x-j;
								minMAD = MAD;
								if(minMAD==0)
									break;
							}
							searchB.release();
						}
					}
				}
			}
			startB.release();
			minMAD = 256;
		}
	}
}

float maxmag(int **Mx, int **My, int patchx, int patchy)
{
	float max = 0;
	float mag;
	int i,j;

	for(i=0;i<patchy;i++)
		for(j=0;j<patchx;j++)
		{
			mag = (float)sqrt(pow(Mx[i][j],2.0)+pow(My[i][j],2.0));
			if(mag > max)
				max = mag;
		}
	return max;
}

Mat DrawGradients(int **My, int **Mx, int cellsize, Mat frame)
{
	double l_max = -cellsize;
	int dx, dy;
	Mat flow;

	frame.copyTo(flow);

	for (int y = 0; y < frame.rows; y+=16)      // First iteration, to compute the maximum l (longest flow)
	{
		for (int x = 0; x < frame.cols; x+=16)
		{
			dx = Mx[y/cellsize][x/cellsize];  
			dy = My[y/cellsize][x/cellsize];   
			double l = sqrt(dx^2 + dy^2);       // This function sets a basic threshold for drawing on the image
			if(l>l_max) l_max = l;
		}
	}


	for (int y = 0; y < frame.rows; y+=cellsize)
	{
		for (int x = 0; x < frame.cols; x+=cellsize)
		{
			rectangle(flow, Point(x,y), Point(x+cellsize,y+cellsize), 127, 1);
			dx = Mx[y/cellsize][x/cellsize];  
			dy = My[y/cellsize][x/cellsize];   
	
			CvPoint p = cvPoint(x, y);

			double l = sqrt(dx*dx + dy*dy);       // This function sets a basic threshold for drawing on the image
			if (l > 0)
			{
				double spinSize = l/2;   // Factor to normalise the size of the spin depeding on the length of the arrow

				CvPoint p2 = cvPoint(p.x + (int)(dx), p.y + (int)(dy));
				line(flow, p, p2, 127, 1, CV_AA);

				double angle;    // Draws the spin of the arrow
				angle = atan2( (double) p.y - p2.y, (double) p.x - p2.x );

				p.x = (int) (p2.x + spinSize * cos(angle + 3.1416 / 4));
				p.y = (int) (p2.y + spinSize * sin(angle + 3.1416 / 4));
				line(flow, p, p2, 127, 1, CV_AA, 0);

				p.x = (int) (p2.x + spinSize * cos(angle - 3.1416 / 4));
				p.y = (int) (p2.y + spinSize * sin(angle - 3.1416 / 4));
				line(flow, p, p2, 127, 1, CV_AA, 0);
			}
		}
	}
	return flow;
}

int main( int argc, char** argv )
{
	VideoCapture cap; // open the video camera for reading (use a string for a file)
	Mat frame, gray_frame;
	Mat prev,curr;
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
	int t,i,j,k,N=15;	
	int WIDTH,HEIGHT,cellsize = 16;
	int patch_x;
	int patch_y;
	int **MVx,**MVy;
	float p;
	float ***M;
	float ***theta;
	float **C,**Ment;
	float s1,s2,MaxMag;

	namedWindow("Coherency Based STSM", CV_WINDOW_AUTOSIZE);
	//namedWindow("Optical Flow",1);

	cap.open(0);

	cap >> frame;

	WIDTH = frame.cols; //640 on hp-pavillion webcam
	HEIGHT = frame.rows; //480 on hp-pavillion webcam
	patch_x = WIDTH/cellsize; //40 on hp-pavillion webcam
	patch_y = HEIGHT/cellsize; //30 on hp-pavillion webcam
	
	C = new float*[patch_y];
	for(i = 0; i < patch_y; ++i)
		C[i] = new float[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			C[i][j] = 0;
	
	Ment = new float*[patch_y];
	for(i = 0; i < patch_y; ++i)
		Ment[i] = new float[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			Ment[i][j] = 0;

	M = new float**[patch_y];
	for(i = 0; i < patch_y; i++)
	{
		M[i] = new float*[patch_x];
		for(j = 0; j < patch_x; j++)
		{
			M[i][j] = new float[14];
		}
	}

	theta = new float**[patch_y];
	for(i = 0; i < patch_y; i++)
	{
		theta[i] = new float*[patch_x];
		for(j = 0; j < patch_x; j++)
		{
			theta[i][j] = new float[14];
		}
	}

	MVx = new int*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MVx[i] = new int[patch_x];
	
	MVy = new int*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MVy[i] = new int[patch_x];
	
	/*************************************************************************/
	/*prev = Mat::zeros(480,640, CV_8UC1);
	next = Mat::zeros(480,640, CV_8UC1);
	for(i=0;i<=15;i++)
		for(j=0;j<=15;j++)
		{
			prev.data[WIDTH*i + j] = 255;
			next.data[WIDTH*(i+4) + (j+4)] = 255;
			prev.data[WIDTH*(i+64) + (j+64)] = 255;
			next.data[WIDTH*(i+64) + (j+69)] = 255;
		}
	ARPS(prev, next, 16, 3, MVx, MVy, 8);
	flow = DrawGradients(MVy,MVx,16, next);
	imshow("Optical Flow", flow);*/
	/**************************************************************************/
	while(1)
    	{
		for(t=0;t<N;t++)
		{
			cap >> frame; // read a new frame from video			
			
			cvtColor(frame, gray_frame, CV_BGR2GRAY);

			gray_frame.convertTo(curr, -1, 1.2, 0);
			
			if(t > 0)
			{
				ARPS(prev, curr, 16, 3, MVx, MVy, 8);
				MaxMag = maxmag(MVx,MVy,patch_x,patch_y);
				if(MaxMag==0)
					MaxMag=1;
				for(i=0;i<patch_y;i++)
					for(j=0;j<patch_x;j++)
					{
						M[i][j][t-1] = (float)sqrt(pow(MVx[i][j],2.0)+pow(MVy[i][j],2.0))/MaxMag;
						theta[i][j][t-1] = (float)atan2((double)MVy[i][j],(double)MVx[i][j]);
					}
			}
			
			if(t == 14)
			{
				d.compute(curr, descriptorsValues, Size(0,0), Size(0,0), locations);
		
				for(i=0;i<patch_y;i++)
				{
					for(j=0;j<patch_x;j++)
					{
						s1 = 0;
						s2 = 0;
						for(k=0;k<9;k++)
							s1 += descriptorsValues[(j * patch_y + i) * 9 + k];
						for(k=0;k<9;k++)
						{
							p = descriptorsValues[(j * patch_y + i) * 9 + k]/s1;
							C[i][j] += p*log10(p);
						}
						C[i][j] = -C[i][j];

						for(k=0;k<t;k++)
						{
							s2 += M[i][j][k];
						}
						for(k=0;k<t;k++)
						{
							p = M[i][j][k]/s2;
							if(s2 == 0 || p == 0)
								Ment[i][j] += 0;
							else
								Ment[i][j] += p*log10(p);
						}
						Ment[i][j] = -Ment[i][j];
					}
				}
				cout<<Ment[0][19]<<endl;
			}
			/*Swap current frame with the previous for next step*/
			swap(prev, curr);
		}
		
		//SM = formula della mappa SM;
		for (int y = 0; y < frame.rows; y+=cellsize)
			for (int x = 0; x < frame.cols; x+=cellsize)
				rectangle(frame, Point(x,y), Point(x+cellsize,y+cellsize), 127, 1);
		circle(frame, Point((19*cellsize+cellsize/2),(0*cellsize)+cellsize/2), 7, 127, 2, 8, 0);

		imshow("Coherency Based STSM",frame);

		for(i=0;i<patch_y;i++)
			for(j=0;j<patch_x;j++)
			{
				C[i][j] = 0;
				Ment[i][j] = 0;
			}
		
		if(waitKey(10) == 27) //wait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
		{
			system("PAUSE");
			cout << "esc key is pressed by user" << endl; 
            break; 
		}
    	}
}
