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


void BMA(Mat A, Mat B, int blocksize)
{
	Mat flow = Mat::zeros(A.rows/blocksize,A.cols/blocksize, A.type());
	const int patch_x = A.cols/blocksize;
	const int patch_y = A.rows/blocksize;
	int WIDTH = A.cols;
	int HEIGHT = A.rows;
	int i,j,dx,dy,n1,n2;
	float MAD = 0;
	float minMAD = 255;
	float **MV_x;
	float **MV_y;
	int w=10;
	int border_l,border_u,border_r,border_d;
	int center_x,center_y;
	int delt_cent_x, delt_cent_y,next_add_x,next_add_y,new_x,new_y;
	
	MV_x = new float*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MV_x[i] = new float[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			MV_x[i][j] = 0;

	MV_y = new float*[patch_y];
	for(i = 0; i < patch_y; ++i)
		MV_y[i] = new float[patch_x];
	for(i=0;i<patch_y;i++)
		for(j=0;j<patch_x;j++)
			MV_y[i][j] = 0;

	/*Start compting motion vectors for each block of the frame*/
	for(dy=0; dy<patch_y; dy++)
		for(dx=0; dx<patch_x; dx++)
		{
			center_y = (blocksize/2)+(blocksize)*dy;
			center_x = (blocksize/2)+(blocksize)*dx;
			new_x = center_x;
			new_y = center_y;
			delt_cent_x = 0;
			delt_cent_y = 0;
			minMAD = 255;

			while(w>4)
			{
				border_r = new_x+(w/3)+(blocksize/2);
				border_l = new_x-(w/3)-(blocksize/2);
				border_u = new_y-(w/3)-(blocksize/2);
				border_d = new_y+(w/3)+(blocksize/2);
							
				if(w==10)
				{
					for(n1=center_y-blocksize/2; n1<center_y+blocksize/2; n1++)
						for(n2=center_x-blocksize/2; n2<center_x+blocksize/2; n2++)
							MAD += abs(A.data[WIDTH*n1 + n2]-B.data[WIDTH*n1 + n2]);
					MAD = MAD/(blocksize*blocksize);	
					minMAD=MAD;
					MAD = 0;
				}

				next_add_x = 0;
				next_add_y = 0;

				if(border_r<WIDTH)
				{
					for(n1=center_y-blocksize/2; n1<center_y+blocksize/2; n1++)
						for(n2=center_x-blocksize/2; n2<center_x+blocksize/2; n2++)
						{
							i = delt_cent_y;
							j = delt_cent_x + (w/3);
							MAD += abs(A.data[WIDTH*(n1+i) + (n2+j)]-B.data[WIDTH*n1 + n2]);
						}
					MAD = MAD/(blocksize*blocksize);	
					if(MAD<minMAD)
					{
						minMAD=MAD;
						next_add_x = (w/3);
						next_add_y = 0;
					}
					MAD = 0;
				}

				if(border_l>=0)
				{
					for(n1=center_y-blocksize/2; n1<center_y+blocksize/2; n1++)
						for(n2=center_x-blocksize/2; n2<center_x+blocksize/2; n2++)
						{
							i = delt_cent_y;
							j = delt_cent_x-(w/3);
							MAD += abs(A.data[WIDTH*(n1+i) + (n2+j)]-B.data[WIDTH*n1 + n2]);
						}
					MAD = MAD/(blocksize*blocksize);	
					if(MAD<minMAD)
					{
						minMAD=MAD;
						next_add_x = -(w/3);
						next_add_y = 0;
					}
					MAD = 0;
				}

				if(border_u>=0)
				{
					for(n1=center_y-blocksize/2; n1<center_y+blocksize/2; n1++)
						for(n2=center_x-blocksize/2; n2<center_x+blocksize/2; n2++)
						{
							i = delt_cent_y-(w/3);
							j = delt_cent_x;
							MAD += abs(A.data[WIDTH*(n1+i) + (n2+j)]-B.data[WIDTH*n1 + n2]);
						}
					MAD = MAD/(blocksize*blocksize);	
					if(MAD<minMAD)
					{
						minMAD=MAD;
						next_add_x = 0;
						next_add_y = -(w/3);
					}
					MAD = 0;
				}

				if(border_d<HEIGHT)
				{
					for(n1=center_y-blocksize/2; n1<center_y+blocksize/2; n1++)
						for(n2=center_x-blocksize/2; n2<center_x+blocksize/2; n2++)
						{
							i = delt_cent_y+(w/3);
							j = delt_cent_x;
							MAD += abs(A.data[WIDTH*(n1+i) + (n2+j)]-B.data[WIDTH*n1 + n2]);
						}
					MAD = MAD/(blocksize*blocksize);	
					if(MAD<minMAD)
					{
						minMAD=MAD;
						next_add_x = 0;
						next_add_y = (w/3);
					}
					MAD = 0;
				}
				/*Change window size*/
				if(next_add_x==0 && next_add_y==0) //no more minMAD found
					w = 0;
				else{
					w = (w/2)+1;
					delt_cent_x += next_add_x;
					delt_cent_y += next_add_y;
					new_x = center_x + delt_cent_x;
					new_y = center_y + delt_cent_y;
				}
			}
			/*Calcola gli 8 intorni di new_x,new_y*/
			border_r = new_x+1+(blocksize/2);
			border_l = new_x-1-(blocksize/2);
			border_u = new_y-1-(blocksize/2);
			border_d = new_y+1+(blocksize/2);
			if(border_u<0)
				border_u = 0;
			else
				border_u = -1;
			if(border_d>=HEIGHT)
				border_d = 0;
			else
				border_d = 1;
			if(border_l<0)
				border_l = 0;
			else
				border_l = -1;
			if(border_r>=WIDTH)
				border_r = 0;
			else
				border_r = 1;

			for(int k = border_u; k<=border_d; k++)
				for(int l = border_l; l<=border_r; l++)
				{
					if(k!=0 || l!=0){
						for(n1=center_y-blocksize/2; n1<center_y+blocksize/2; n1++)
							for(n2=center_x-blocksize/2; n2<center_x+blocksize/2; n2++)
							{
								i = delt_cent_y+k;
								j = delt_cent_x+l;
								MAD += abs(A.data[WIDTH*(n1+i) + (n2+j)]-B.data[WIDTH*n1 + n2]);
							}
						MAD = MAD/(blocksize*blocksize);
						if(MAD<minMAD)
						{
							minMAD = MAD;
							MV_y[dy][dx] = i;
							MV_x[dy][dx] = j;
						}
					}
				}
		}

		for(dy=0; dy<patch_y; dy++)
		{
			for(dx=0; dx<patch_x; dx++)
				cout << "("<<MV_x[dy][dx]<<","<<MV_y[dy][dx]<<") ";
			cout << endl;
		}
		system("PAUSE");
}


int main( int argc, char** argv )
{
	VideoCapture cap; // open the video camera for reading (use a string for a file)
	Mat frame, gray_frame;
	Mat prev,next;
	Mat bgr;//CV_32FC3 matrix
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
	float ***p;
	float **C;
	float s;
	
	//namedWindow("Coherency Based Spatio-Temporal SM (Up to 15 frames)", CV_WINDOW_AUTOSIZE);
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

	while(1)
    {
		//for(N=0;N<15;N++)
		//{
			cap >> frame; // read a new frame from video			
			
			cvtColor(frame, gray_frame, CV_BGR2GRAY);

			gray_frame.convertTo(next, -1, 1.2, 0);
			
			if(N > 0)
			{
				BMA(prev, next, 5);
				imshow("Optical Flow",frame);
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

		//imshow("Coherency Based Spatio-Temporal SM (Up to 15 frames)",SM);

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
