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
					if(minMAD<thres)
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
								if(minMAD<thres)
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
	String file_name = "parachute.avi";
	Rect im_resize = Rect(0,0,408,352); /*parachute*/
	//Rect im_resize = Rect(0,0,256,320); /*birdfall*/
	//Rect im_resize = Rect(0,0,640,480);
	VideoCapture cap(file_name); // open the video camera for reading (use a string for a file)
	Mat frame, gray_frame;
	Mat prev,curr,tmp,tmp1;
	Mat flow;
	int t = 0,i,j,k;	
	int WIDTH,HEIGHT,cellsize = 8;
	int patch_x;
	int patch_y;
	int **MVx,**MVy;

	namedWindow("Optical Flow",1);

	cap >> frame;
	frame = frame(im_resize);
	cvtColor(frame, gray_frame, CV_BGR2GRAY);
	equalizeHist( gray_frame, tmp );
	GaussianBlur(tmp,tmp1,Size(0,0),3);
	addWeighted(tmp, 1.5, tmp1, -0.5, 0, curr);
	
	WIDTH = frame.cols; //640 on hp-pavillion webcam
	HEIGHT = frame.rows; //480 on hp-pavillion webcam
	patch_x = WIDTH/cellsize; //40 on hp-pavillion webcam
	patch_y = HEIGHT/cellsize; //30 on hp-pavillion webcam
	
	MVx = new int*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MVx[i] = new int[patch_x];
	
	MVy = new int*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MVy[i] = new int[patch_x];
		
	while(1)
    	{
			
		if(t > 0)
		{
			ARPS(prev, curr, cellsize, 5, MVx, MVy, 16);
			flow = DrawGradients(MVy, MVx, cellsize, frame);
			imshow("Optical Flow", flow);
		}
		
		/*Swap current frame with the previous for next step*/
		swap(prev, curr);

		cap >> frame; // read a new frame from video			
		if(frame.empty())
		{
			cap = VideoCapture(file_name);
			cap >> frame;
		}

		frame = frame(im_resize);
		cvtColor(frame, gray_frame, CV_BGR2GRAY);
		equalizeHist( gray_frame, tmp );
		GaussianBlur(tmp,tmp1,Size(0,0),3);
		addWeighted(tmp, 1.5, tmp1, -0.5, 0, curr);

		t = 1;

		if(waitKey(10) == 27) //wait for 'esc' key press for 10 ms. If 'esc' key is pressed, break loop
		{
			system("PAUSE");
			cout << "esc key is pressed by user" << endl; 
            		break; 
		}
	}
}
