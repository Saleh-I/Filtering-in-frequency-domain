#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

void fftshift(const Mat &input_img, Mat &output_img)
{
	output_img = input_img.clone();
	int cx = output_img.cols / 2;
	int cy = output_img.rows / 2;
	Mat q1(output_img, Rect(0, 0, cx, cy));
	Mat q2(output_img, Rect(cx, 0, cx, cy));
	Mat q3(output_img, Rect(0, cy, cx, cy));
	Mat q4(output_img, Rect(cx, cy, cx, cy));

	Mat temp;
	q1.copyTo(temp);
	q4.copyTo(q1);
	temp.copyTo(q4);
	q2.copyTo(temp);
	q3.copyTo(q2);
	temp.copyTo(q3);
}


void calculateDFT(Mat &scr, Mat &dst)
{
	// define mat consists of two mat, one for real values and the other for complex values
	Mat planes[] = { scr, Mat::zeros(scr.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);
	dst = complexImg;
}

Mat construct_H(Mat &scr, String type, float D0)
{
	Mat H(scr.size(), CV_32F, Scalar(1));
	float D = 0;
	if (type == "Ideal")
	{
		for (int u = 0; u < H.rows; u++)
		{
			for (int  v = 0; v < H.cols; v++)
			{
				D = sqrt((u - scr.rows / 2)*(u - scr.rows / 2) + (v - scr.cols / 2)*(v - scr.cols / 2));
				if (D > D0)
				{
					H.at<float>(u, v) = 0;
				}
			}
		}
		return H;
	}
	else if (type == "Gaussian")
	{
		for (int  u = 0; u < H.rows; u++)
		{
			for (int v = 0; v < H.cols; v++)
			{
				D = sqrt((u - scr.rows / 2)*(u - scr.rows / 2) + (v - scr.cols / 2)*(v - scr.cols / 2));
				H.at<float>(u, v) = exp(-D*D / (2 * D0*D0));
			}
		}
		return H;
	}
}


void filtering(Mat &scr, Mat &dst, Mat &H)
{
	fftshift(H, H);
	Mat planesH[] = { Mat_<float>(H.clone()), Mat_<float>(H.clone()) };

	Mat planes_dft[] = { scr, Mat::zeros(scr.size(), CV_32F) };
	split(scr, planes_dft);

	Mat planes_out[] = { Mat::zeros(scr.size(), CV_32F), Mat::zeros(scr.size(), CV_32F) };
	planes_out[0] = planesH[0].mul(planes_dft[0]);
	planes_out[1] = planesH[1].mul(planes_dft[1]);

	merge(planes_out, 2, dst);

}


int main()
{
	Mat imgIn = imread("car.png", 0);
	imgIn.convertTo(imgIn, CV_32F);

	// DFT
	Mat DFT_image;
	calculateDFT(imgIn, DFT_image);

	// construct H
	Mat H;
	H = construct_H(imgIn, "Ideal", 85);

	// filtering
	Mat complexIH;
	filtering(DFT_image, complexIH, H);

	// IDFT
	Mat imgOut;
	dft(complexIH, imgOut, DFT_INVERSE | DFT_REAL_OUTPUT);

	normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);

	imshow("img", imgIn);
	imshow("DFT", imgOut);
	waitKey(0);
	return 0;
}

