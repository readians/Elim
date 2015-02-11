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
#define eu 2.7182818284

float costFunMAD(Mat curr, Mat ref, int n){
	int i,j;
	float err = 0;

	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			err = err + abs(curr.data[curr.cols*i + j] - ref.data[ref.cols*i + j]);

	return err / (n*n);
}

void ARPS(Mat A, Mat B, int blocksize, int p, int **MVx, int **MVy, double thres)
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
					T = 2;
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
	String file_name = "birdfall2.avi";
	VideoCapture cap(file_name); // open the video (0 for webcam, string for a file)
	Mat input, frame, prev, curr, gray_frame;
	Mat fgmask,fgimg,bgimg,tmp,tmp1,canny_out,gradx,grady;
	Mat SM;
	//Rect im_resize = Rect(0,0,640,480); /*Cam*/
	Rect im_resize = Rect(0,0,256,320); /*birdfall2*/
	//Rect im_resize = Rect(0,0,408,352); /*parachute*/
	//Rect im_resize = Rect(0,0,400,320); /*girl*/
	vector<float> descriptorsValues;
	vector<Point> locations;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	int t = 0,i,j,k,l,N=15,fr;	
	int WIDTH, HEIGHT, cellsize = 8;
	int patch_x;
	int patch_y;
	int **MVx,**MVy;
	bool flag = false;
	double w = 1, SM_thresh = 0, ARPS_thresh = 16;
	double p,k_e = 0.2;
	double ***M;
	double ***theta;
	double **C,**Ment,**Mcs,**Dent,**Dcs;
	double s,MaxMag,MTemp,thetaTemp;
	
	namedWindow("CSM",CV_WINDOW_AUTOSIZE);
	namedWindow("Current Frame",CV_WINDOW_AUTOSIZE);
	namedWindow("Curr",CV_WINDOW_AUTOSIZE);
	/*Read new frame*/
	cap >> input;
	/*Adjust image input*/
	frame = input(im_resize);
	cvtColor(frame, gray_frame, CV_BGR2GRAY);
	equalizeHist( gray_frame, tmp );
	tmp.convertTo(curr, -1, 1, 0);
	/*Set HOG descriptor*/
	HOGDescriptor d(curr.size(),Size(cellsize,cellsize),Size(cellsize,cellsize),Size(cellsize,cellsize),9,0,-1,HOGDescriptor::L2Hys, 0.2, false, HOGDescriptor::DEFAULT_NLEVELS);
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
	/*-----------------------------------Instantiate variables according to frame dimensions ----------------------*/
	WIDTH = frame.cols; 
	HEIGHT = frame.rows; 
	patch_x = WIDTH/cellsize; 
	patch_y = HEIGHT/cellsize; 
	
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

	SM = Mat::zeros(HEIGHT,WIDTH, CV_8UC1);
	/*-----------------------------------Variables defination complete---------------------------------------*/
	while(1)
	{	
		if(!flag)	
			fr = max(0,t-1);
		else
			fr = (fr+1)%(N-1);

		/*Computing M and theta for each block of the frame with respect the previous frame t-1*/
		if(t > 0 || flag == true)
		{	
			ARPS(prev, curr, cellsize, 6, MVx, MVy, ARPS_thresh);
			MaxMag = maxmag(MVx,MVy,patch_x,patch_y);
			if(MaxMag==0)
				MaxMag=1;
			for(i=0;i<patch_y;i++)
				for(j=0;j<patch_x;j++)
				{
					M[i][j][fr] = sqrt(pow(MVx[i][j],2.0)+pow(MVy[i][j],2.0))/MaxMag;
					if(MVx[i][j] != 0 || MVy[i][j] != 0)	
						theta[i][j][fr] = atan2(MVy[i][j],MVx[i][j]);
					else
						theta[i][j][fr] = 0;
					if(theta[i][j][fr]<0)
						theta[i][j][fr] += (2*PI);
					theta[i][j][fr] /= (2*PI);
				}
		}/*End if*/

		if(t == (N-1) || flag == true) // Compute maps
		{
			/*Set first cycle flag complete*/
			flag = true;
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
						//s += pow(eu,-(descriptorsValues[(j * patch_y + i) * 9 + k]/k_e));
					if(s!=0)
						for(k=0;k<9;k++)
						{
							p = descriptorsValues[(j * patch_y + i) * 9 + k]/s;
							//p = pow(eu,-(descriptorsValues[(j * patch_y + i) * 9 + k]/k_e))/s;
							if(p == 0)
								continue;
							else
								C[i][j] += p*(log10(p));
						}
					C[i][j] = -C[i][j];

					/*Motion Entropy Map for block(i,j)*/
					s = 0;
					for(k=0;k<N-1;k++)
						//s += pow(eu,-(M[i][j][k]/k_e));
						s += M[i][j][k];
					if(s!=0)
						for(k=0;k<N-1;k++)
						{
							//p = (pow(eu,-(M[i][j][k]/k_e))/s);
							p = M[i][j][k]/s;
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
								MTemp = M[i+k][j+l][fr];
							Mcs[i][j]+=abs(M[i][j][fr]-MTemp);
						}
					Mcs[i][j] = Mcs[i][j]/8;

					/*Direction Entropy Map for block(i,j)*/
					s = 0;
					for(k=0;k<N-1;k++)
						//s += pow(eu,-(theta[i][j][k]/k_e));
						s += theta[i][j][k];
					if(s!=0)
						for(k=0;k<N-1;k++)
						{
							//p = (pow(eu,-(theta[i][j][k]/k_e))/s);
							p = theta[i][j][k]/s;
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
								thetaTemp = theta[i+k][j+l][fr];
							Dcs[i][j]+=abs(theta[i][j][fr]-thetaTemp);
						}
					Dcs[i][j] = Dcs[i][j]/8;
						
					/*Final Spatio Temporal Saliency Map for block(i,j)*/
					s = (w*(1-C[i][j])) + ((1-w)*((Ment[i][j]*Mcs[i][j])+((1-Dent[i][j])*Dcs[i][j])));
					/*Zero Thresholding*/
					if(s<SM_thresh)
						s = 0;
					for(k=i*cellsize;k<(i*cellsize)+cellsize;k++)
						for(l=j*cellsize;l<(j*cellsize)+cellsize;l++)
							SM.data[SM.cols*k+l] = (uchar)(s*255); 
				}
			}
		}/*End if*/	
		
		/*Find and draw contours*/
		Canny(SM, canny_out, 0, 255, 3);//with or without, explained later.
		findContours(canny_out, contours, hierarchy, CV_RETR_EXTERNAL, 2, Point(0,0));
		for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
		{
			if (it->size()<50)
				it=contours.erase(it);
			else
				++it;
		}
		for(i = 0; i<contours.size(); i++ )
		{
			Scalar color = Scalar(0, 255, 0);
			drawContours(frame, contours, i, color, 2, 8, hierarchy, 0, Point());
		}

		/*Show result*/
		imshow("CSM",SM);
		imshow("Current Frame",frame);
		imshow("Curr",curr);
		/*Swap current frame with the previous for next step*/
		swap(prev, curr);
		/*Read new frame*/
		cap >> input;
		if(input.empty())
		{
			cap = VideoCapture(file_name);
			cap >> input;
			frame = input(im_resize);
			cvtColor(frame, gray_frame, CV_BGR2GRAY);
			equalizeHist( gray_frame, tmp );
			tmp.convertTo(curr, -1, 1, 0);
			t = 0;
			fr = 0;
			flag = false;
			for(i=0;i<HEIGHT;i++)
				for(j=0;j<WIDTH;j++)
					SM.data[SM.cols*i+j] = 0;
		}
		else
		{
			frame = input(im_resize);
			cvtColor(frame, gray_frame, CV_BGR2GRAY);
			equalizeHist( gray_frame, tmp );
			tmp.convertTo(curr, -1, 1, 0);
		}
		/*Circular index*/
		t = (t+1)%N;
		/*Reset Maps*/
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
    }/*End While(1)*/
}
