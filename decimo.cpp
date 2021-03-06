#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
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
				if(j==0 || (MVx[patch_y][patch_x-1]==0 && MVy[patch_y][patch_x-1]==0))
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
					if(minMAD < thres)
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
								if(minMAD < thres)
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


int main( int argc, char** argv )
{
	VideoCapture cap; // open the video camera for reading (use a string for a file)
	Mat frame, gray_frame;
	Mat prev,curr;
	Mat canny_out;
	Mat SM,SM8U;
	vector<float> descriptorsValues;
	vector<Point> locations;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	int t = 0,i,j,k,l,N=15;	
	int WIDTH,HEIGHT,cellsize = 8;
	int patch_x;
	int patch_y;
	int **MVx,**MVy;
	double w = 0.3;
	double p;
	double ***M;
	double ***theta;
	double **C,**Ment,**Mcs,**Dent,**Dcs;
	double s,MaxMag,MTemp,thetaTemp;
	HOGDescriptor d(Size(640,480),Size(cellsize,cellsize),Size(cellsize,cellsize),Size(cellsize,cellsize),9,0,-1,HOGDescriptor::L2Hys, 0.2, false, HOGDescriptor::DEFAULT_NLEVELS);
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
	/*Set window for the output map*/
	namedWindow("Coherency Based STSM", CV_WINDOW_AUTOSIZE);
	namedWindow("prova",CV_WINDOW_AUTOSIZE);

	/*Open webcam*/
	cap.open(0);
	/*Read new frame*/
	cap >> frame;

	/*-----------------------------------Define variables according to frame dimensions ----------------------*/
	WIDTH = frame.cols; //640 on hp-pavillion webcam
	HEIGHT = frame.rows; //480 on hp-pavillion webcam
	patch_x = WIDTH/cellsize; //40 on hp-pavillion webcam
	patch_y = HEIGHT/cellsize; //30 on hp-pavillion webcam
	
	C = new double*[patch_y];
	for(i = 0; i < patch_y; ++i)
		C[i] = new double[patch_x];
	
	Ment = new double*[patch_y];
	for(i = 0; i < patch_y; ++i)
		Ment[i] = new double[patch_x];

	Mcs = new double*[patch_y];
	for(i = 0; i < patch_y; ++i)
		Mcs[i] = new double[patch_x];

	Dent = new double*[patch_y];
	for(i = 0; i < patch_y; ++i)
		Dent[i] = new double[patch_x];

	Dcs = new double*[patch_y];
	for(i = 0; i < patch_y; ++i)
		Dcs[i] = new double[patch_x];

	M = new double**[patch_y];
	for(i = 0; i < patch_y; i++)
	{
		M[i] = new double*[patch_x];
		for(j = 0; j < patch_x; j++)
		{
			M[i][j] = new double[N-1];
		}
	}

	theta = new double**[patch_y];
	for(i = 0; i < patch_y; i++)
	{
		theta[i] = new double*[patch_x];
		for(j = 0; j < patch_x; j++)
		{
			theta[i][j] = new double[N-1];
		}
	}

	MVx = new int*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MVx[i] = new int[patch_x];
	
	MVy = new int*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MVy[i] = new int[patch_x];
	
	/*Initialization*/
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
		{
			C[i][j] = 0;
			Ment[i][j] = 0;
			Dent[i][j] = 0;
			Mcs[i][j] = 0;
			Dcs[i][j] = 0;
		}

	SM = Mat::zeros(HEIGHT,WIDTH, CV_64F);
	SM8U = Mat::zeros(HEIGHT,WIDTH, CV_8UC1);
	/*-----------------------------------Variables defination complete---------------------------------------*/
	while(1)
    {
		for(t=0;t<N;t++) //Start computing saliency map (up to N frames)
		{
			cap >> frame; //Read a new frame from video			
			
			cvtColor(frame, gray_frame, CV_BGR2GRAY); //Change to grayscale

			gray_frame.convertTo(curr, -1, 1, 0); //"curr" is the current frame
			
			/*Computing M and theta for each block of the frame with respect the previous frame t-1*/
			if(t > 0)
			{
				ARPS(prev, curr, cellsize, 3, MVx, MVy, 8);
				MaxMag = maxmag(MVx,MVy,patch_x,patch_y);
				if(MaxMag==0)
					MaxMag=1;
				for(i=0;i<patch_y;i++)
					for(j=0;j<patch_x;j++)
					{
						M[i][j][t-1] = (double)sqrt(pow(MVx[i][j],2.0)+pow(MVy[i][j],2.0))/MaxMag;
						if(MVx[i][j] != 0 || MVy[i][j] != 0)	
							theta[i][j][t-1] = atan2(MVy[i][j],MVx[i][j]);
						else
							theta[i][j][t-1] = 0;
						if(theta[i][j][t-1]<0)
							theta[i][j][t-1] += (2*PI);
						theta[i][j][t-1] /= (2*PI);
					}
			}

			if(t == (N-1)) // Compute maps
			{
				/*Compute HOG for the current frame*/
				d.compute(curr, descriptorsValues, Size(0,0), Size(0,0), locations);
				for(i=0;i<patch_y;i++)
				{
					for(j=0;j<patch_x;j++)
					{
						/*Spatial Coherency Map for block(i,j)*/
						s = 0;
						for(k=0;k<9;k++)
							s += descriptorsValues[(j * patch_y + i) * 9 + k];
						if(s!=0)
							for(k=0;k<9;k++)
							{
								p = descriptorsValues[(j * patch_y + i) * 9 + k]/s;
								if(p == 0)
									continue;
								else
									C[i][j] += p*log10(p);
							}
						C[i][j] = -C[i][j];
						C[i][j] = (1 - C[i][j]);

						/*Motion Entropy Map for block(i,j)*/
						s = 0;
						for(k=0;k<t;k++)
							s += M[i][j][k];
						if(s!=0)
							for(k=0;k<t;k++)
							{
								p = (M[i][j][k]/s);
								if(p == 0)
									continue;
								else
									Ment[i][j] += p*log10(p);
							}
						Ment[i][j] = -Ment[i][j];
						
						/*Motion CS Map for block(i,j)*/
						for(k=-1;k<=1;k++)
							for(l=-1;l<=1;l++)
							{
								if(i+k<0 || j+l<0 || i+k>=patch_y || j+l>=patch_x)
									MTemp = 0;
								else
									MTemp = M[i+k][j+l][t-1];
								Mcs[i][j]+=abs(M[i][j][t-1]-MTemp);
							}
						Mcs[i][j] = Mcs[i][j]/8;

						/*Direction Entropy Map for block(i,j)*/
						s = 0;
						for(k=0;k<t;k++)
							s += theta[i][j][k];
						if(s!=0)
							for(k=0;k<t;k++)
							{
								p = (theta[i][j][k]/s);
								if(p == 0)
									continue;
								else
									Dent[i][j] += p*log10(p);
							}
						Dent[i][j] = -Dent[i][j];
						
						/*Direction CS Map for block(i,j)*/
						for(k=-1;k<=1;k++)
							for(l=-1;l<=1;l++)
							{
								if(i+k<0 || j+l<0 || i+k>=patch_y || j+l>=patch_x)
									thetaTemp = 0;
								else
									thetaTemp = theta[i+k][j+l][t-1];
								Dcs[i][j]+=abs(theta[i][j][t-1]-thetaTemp);
							}
						Dcs[i][j] = Dcs[i][j]/8;
						
						/*Final Spatio Temporal Saliency Map for block(i,j)*/
						s = (w*(1-C[i][j]) + (1-w)*((Ment[i][j]*Mcs[i][j])+((1-Dent[i][j])*Dcs[i][j])));
						/*Zero Thresholding*/
						if(s<0.3)
							s = 0;
						for(k=i*cellsize;k<(i*cellsize)+cellsize;k++)
							for(l=j*cellsize;l<(j*cellsize)+cellsize;l++)
							{
								SM.at<double>(k,l) = s;
								SM8U.data[WIDTH*k+l] = (uchar)(s*255); 
							}
					}
				}
			}
			/*Swap current frame with the previous for next step*/
			swap(prev, curr);
		}	
		
		for (int y = 0; y < frame.rows; y+=cellsize)
			for (int x = 0; x < frame.cols; x+=cellsize)
				rectangle(frame, Point(x,y), Point(x+cellsize,y+cellsize), 127, 1);
		circle(frame, Point((39*cellsize+cellsize/2),(1*cellsize)+cellsize/2), 7, 127, 2, 8, 0);

		cout<<C[1][39]<<"\t"<<Ment[1][39]<<"\t"<<Dent[1][39]<<"\t"<<Mcs[1][39]<<"\t"<<Dcs[1][39]<<endl;

		//Canny(SM8U,canny_out,80,160,3);
		findContours(SM8U, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		for(i = 0; i< contours.size(); i++ )
		{
			Scalar color = Scalar(0, 255, 0);
			drawContours(frame, contours, i, color, 2, 8, hierarchy, 0, Point());
		}

		/*Show result*/
		imshow("Coherency Based STSM",frame);
		imshow("prova",SM8U);

		/*Reset maps*/
		for(i=0;i<patch_y;i++)
			for(j=0;j<patch_x;j++)
			{
				C[i][j] = 0;
				Ment[i][j] = 0;
				Dent[i][j] = 0;
				Mcs[i][j] = 0;
				Dcs[i][j] = 0;
			}
		
		if(waitKey(10) == 27) //wait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
		{
			system("PAUSE");
			cout << "esc key is pressed by user" << endl; 
            break; 
		}
    }
}
