#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "optidigit.h"
#include "mlp2026uni.h"

double dwoi[L - 1][NMAX][NMAX], dbj[L - 1][NMAX];

void InitW(int ns[], double woi[][NMAX][NMAX], double bj[][NMAX],double beta[][NMAX],double gamma[][NMAX], double mmean[][NMAX], double mvar[][NMAX]) {
	int l, j, i,k;
	for (l = 0; l < L - 1; l++) {
		for (j = 0; j < ns[l + 1]; j++) {
			bj[l][j] = 0;
			for (i = 0; i < ns[l]; i++) {
				woi[l][j][i] = rand() / (RAND_MAX + 1.0) - 0.5;
			}
		}
	}

	for (l = 0; l < L; l++) {
		for (j = 0; j < ns[l + 1]; j++) {
			beta[l][j] = 0, gamma[l][j] = 1;
			mmean[l][j] = 0;
			mvar[l][j] = 1;
		}
	}
}

void Forward(int ns[], double out[][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX], int oact,int hact, double mmean[][NMAX], double mvar[][NMAX], double gamma[][NMAX], double beta[][NMAX]) {
	int l, j, i; 
	double myu,max,sum;

	for (l = 0; l < L - 2; l++) {//hide
		for (j = 0; j < ns[l + 1]; j++) { 
			for (myu= bj[l][j],i = 0; i < ns[l]; i++) {
				myu += out[l][i] * woi[l][j][i];
			}

			if (BATCHLEARN && BATCHNORM) {

				myu=gamma[l][j]* (myu - mmean[l][j]) / sqrt(mvar[l][j] + EPS)+ beta[l][j];

			}



			if (hact == RELU){
				if (myu < 0) out[l + 1][j] = 0;
				else out[l + 1][j] = myu;


			}
			else out[l + 1][j] = 1.0 / (1.0 + exp(-myu));
		}
	}
	//output
	if (oact == SOFTCROSS) {
		j = 0;
		for (myu = bj[l][j], i = 0; i < ns[l]; i++) {
			myu += out[l][i] * woi[l][j][i];
		}
		out[l + 1][j] = myu;
		for (max = out[l + 1][0], j = 1; j < ns[l + 1]; j++) {//output
			for (myu = bj[l][j], i = 0; i < ns[l]; i++) {
				myu += out[l][i] * woi[l][j][i];
			}
			out[l + 1][j] = myu;
			if (max < myu)max = myu;
		}
		for (sum = 0, j = 0; j < ns[l + 1]; j++) {
			out[l + 1][j] = exp(out[l + 1][j] - max);
			sum += out[l + 1][j];
		}
		for (j = 0; j < ns[l + 1]; j++) {
			out[l + 1][j] /= sum;
		}
	
	}
	else { //defoしぐもいど
		
		for (j = 0; j < ns[l + 1]; j++) {
			for (myu = bj[l][j], i = 0; i < ns[l]; i++) {
				myu += out[l][i] * woi[l][j][i];
			}
			out[l + 1][j] = 1.0 / (1.0 + exp(-myu));
		}

		
	}
	
}
void BForward(int ns[], double out[][BS][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX], int oact, int hact,
	double net[][NMAX][BS],double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double mmean[][NMAX], double mvar[][NMAX]) {
	int l, j, i,k,b;
	double myu[BS], max, sum;


	
	for (l = 0; l < L - 2; l++) {//hide
		for (j = 0; j < ns[l + 1]; j++) {

			for (b = 0; b < BS; b++) {
				for (myu[b] = bj[l][j], i = 0; i < ns[l]; i++) {
					myu[b] += out[l][b][i] * woi[l][j][i];
				}
			}
			//batsch noralization
			if (BATCHNORM) {
				for (b = 0; b < BS; b++) net[l][j][b] = myu[b];
				
				BatchNormF(net[l][j], myu, &mean[l][j], &var[l][j], gamma[l][j], beta[l][j]);
				mmean[l][j] = mmean[l][j] * 0.9 + mean[l][j] * 0.1;
				mvar[l][j] = mvar[l][j] * 0.9 + var[l][j] * 0.1;
				
			}
			for (b = 0; b < BS; b++) {
				if (hact == RELU) {
					if (myu[b] < 0) out[l + 1][b][j] = 0;
					else out[l + 1][b][j] = myu[b];


				}
				else out[l + 1][b][j] = 1.0 / (1.0 + exp(-myu[b]));
			}
		}
	}
	
		if (oact == SOFTCROSS) {
			for (b = 0; b < BS; b++) {
				j = 0;
				for (myu[b] = bj[l][j], i = 0; i < ns[l]; i++) {
					myu[b] += out[l][b][i] * woi[l][j][i];
				}
				out[l + 1][b][j] = myu[b];
				max = myu[b];
				for (j = 1; j < ns[l + 1]; j++) {//output
					for (myu[b] = bj[l][j], i = 0; i < ns[l]; i++) {
						myu[b] += out[l][b][i] * woi[l][j][i];
					}
					out[l + 1][b][j] = myu[b];
					
					if (max < myu[b])max = myu[b];
				}

				for (sum = 0, j = 0; j < ns[l + 1]; j++) {
					out[l + 1][b][j] = exp(out[l + 1][b][j] - max);
					sum += out[l + 1][b][j];
				}
				for (j = 0; j < ns[l + 1]; j++) {
					out[l + 1][b][j] /= sum;
				}
			}

		}
		else { //defoしぐもいど
			

			for (j = 0; j < ns[l + 1]; j++) {//output
				for (b = 0; b < BS; b++) {
					for (myu[b] = bj[l][j], i = 0; i < ns[l]; i++) {
						myu[b] += out[l][b][i] * woi[l][j][i];
					}
					out[l + 1][b][j] = 1.0 / (1.0 + exp(-myu[b]));
				}
			}

		}
	

}

void BackProp(int ns[], double out[][NMAX], double Tk[], double delta[][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX],double eta,int oact,int hact) {
	int l, j, i,k;
	double delta_sum = 0.0;
	//修正量計算
	//出力
	l = L - 1; //j=l-1 k=l
	if (oact == SOFTCROSS) {
		for (k = 0; k < ns[l]; k++) {
			delta[l][k] = out[l][k] - Tk[k];//soft cross entoropy bibunn 

			for (j = 0; j < ns[l - 1]; j++) {
				dwoi[l - 1][k][j] = -eta * delta[l][k] * out[l - 1][j];

			}
			dbj[l - 1][k] = -eta * delta[l][k];
		}
	}
	else{

		for (k = 0; k < ns[l]; k++) {
			delta[l][k] = (out[l][k] - Tk[k]) * out[l][k] * (1.0 - out[l][k]);

			for (j = 0; j < ns[l - 1]; j++) {
				dwoi[l - 1][k][j] = -eta * delta[l][k] * out[l - 1][j];

			}
			dbj[l - 1][k] = -eta * delta[l][k];
		}
	}
	//隠れ層のやつ自分
	/*for (l = L - 2; l > 0; l--) {
		for (j = 0; j < ns[l]; j++) {
			delta_sum = 0.0;
			for (int k = 0; k < ns[l + 1]; k++) {
				delta_sum += delta[l + 1][k] * woi[l][k][j];
			}
			delta[l][j] = delta_sum * out[l][j] * (1.0 - out[l][j]);
			dbj[l - 1][j] = -eta * delta[l][j];
			for (i = 0; i < ns[l - 1]; i++) {
				dwoi[l - 1][j][i] = -eta * delta[l][j] * out[l - 1][i];
			}
		}
	}*/
	//先生バージョン
	for (l = L - 2; l > 0; l--) {
		for (j = 0; j < ns[l]; j++) {
			for (delta[l][j]=0, k = 0; k < ns[l + 1]; k++) {
				 delta[l][j]+= delta[l+1][k] * woi[l][k][j];
				 
			}
			if (hact == RELU) {
				if (out[l][j] <= 0) delta[l][j] = 0;
			}
			else delta[l][j] *= out[l][j] * (1.0 - out[l][j]);
			for (i = 0; i < ns[l - 1]; i++) {
				dwoi[l - 1][j][i] = -eta * delta[l][j] * out[l - 1][i];
				
			}
			dbj[l - 1][j] = -eta * delta[l][j];
		}
	}



	//重みとバイアスの更新
	for (l = 0; l < L - 1; l++) {
		for (j = 0; j < ns[l + 1]; j++) {
			bj[l][j] += dbj[l][j];
			/*dbj[l][j] = 0;*/
			for (i = 0; i < ns[l]; i++) {
				woi[l][j][i] += dwoi[l][j][i];
				/*dwoi[l][j][i] = 0;*/

			}
		}
	}

	
}
void BBackProp(int ns[], double out[][BS][NMAX], double Tk[BS][OD], double delta[][NMAX][BS], double woi[][NMAX][NMAX], double bj[][NMAX], double eta, int oact, int hact, double net[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX]) {
	int l, j, i, k,b;
	double delta_sum = 0.0;
	//修正量計算
	//出力がら


	for (l = 0; l < L - 1; l++) {
		for (j = 0; j < ns[l + 1]; j++) {
			dbj[l][j] = 0.0;
			for (i = 0; i < ns[l]; i++) {
				dwoi[l][j][i] = 0.0;
			}
		}
	}

	l = L - 1; //j=l-1 k=l
	if (oact == SOFTCROSS) {
		for (b = 0; b < BS; b++) {
			for (k = 0; k < ns[l]; k++) {
				delta[l][k][b] = out[l][b][k] - Tk[b][k];
				for (j = 0; j < ns[l - 1]; j++) {
					dwoi[l - 1][k][j] += -eta * delta[l][k][b] * out[l - 1][b][j];
				}
				dbj[l - 1][k] += -eta * delta[l][k][b];
			}
		}
	}
	else {
		for (b = 0; b < BS; b++) {
			for (k = 0; k < ns[l]; k++) {
				delta[l][k][b] = (out[l][b][k] - Tk[b][k]) * out[l][b][k] * (1.0 - out[l][b][k]);
				for (j = 0; j < ns[l - 1]; j++) {
					dwoi[l - 1][k][j] += -eta * delta[l][k][b] * out[l - 1][b][j];
				}
				dbj[l - 1][k] += -eta * delta[l][k][b];
			}
		}
		
	}
	for (l = L - 2; l > 0; l--) {
		for (j = 0; j < ns[l]; j++) {
			for (b = 0; b < BS; b++) {

				delta[l][j][b] = 0.0;
				for (k = 0; k < ns[l + 1]; k++) {
					delta[l][j][b] += delta[l + 1][k][b] * woi[l][k][j];
				}
				if (hact == RELU) {
					if (out[l][b][j] <= 0) delta[l][j][b] = 0.0;
				}
				else {
					delta[l][j][b] *= out[l][b][j] * (1.0 - out[l][b][j]);
				}
			}
			if (BATCHNORM) {
				//BN back
				BatchNormB(net[l - 1][j], delta[l][j], mean[l - 1][j], var[l - 1][j], &gamma[l - 1][j], &beta[l - 1][j], eta);
			}

			for (b = 0; b < BS; b++) {
				for (i = 0; i < ns[l - 1]; i++) {
					dwoi[l - 1][j][i] += -eta * delta[l][j][b] * out[l - 1][b][i];
				}
				dbj[l - 1][j] += -eta * delta[l][j][b];

			}
		}
	}




	//重みとバイアスの更新
	for (l = 0; l < L - 1; l++) {
		for (j = 0; j < ns[l + 1]; j++) {
			bj[l][j] += dbj[l][j]/BS;
			for (i = 0; i < ns[l]; i++) {
				woi[l][j][i] += dwoi[l][j][i]/BS;
				

			}
		}
	}


}

void BatchNormF(double net[], double myu[], double* mean, double* var, double gamma, double beta) {
	double sum, ave,sqvar;
	int b;
	for (sum = 0, b = 0; b < BS; b++) sum += net[b];
	ave = sum / BS;
	*mean = ave;
	for (sum = 0, b = 0; b < BS; b++) sum += (net[b] - ave) * (net[b] - ave);
	*var = sum / BS;

	sqvar = sqrt(*var + EPS);

	for (sum = 0, b = 0; b < BS; b++)myu[b] = gamma * ((net[b] - ave) / sqvar) + beta;
}
void BatchNormB(double net[], double delta[], double mean, double var, double* gamma, double* beta, double eta) {
	double dg=0.0, db=0.0;
	double sqvar = sqrt(var + EPS);
	double x_hat[BS], d12a[BS];
	double sum_d = 0.0, sum_xhat_d = 0.0;//yから1個前のやつ
	int b;
	
	for (b = 0; b < BS; b++) {
		x_hat[b] = (net[b] - mean) / sqvar;
		dg += delta[b] * x_hat[b];
		db += delta[b];
	}
	//更新したガンマを使うと先生と一緒になる．
	//*gamma += -eta * dg/BS;
	//*beta += -eta * db / BS;

	for (b = 0; b < BS; b++) {
		d12a[b] = delta[b] * (*gamma);
		sum_d += d12a[b];
		sum_xhat_d += d12a[b] * x_hat[b]; 
	
	}
	for (b = 0; b < BS; b++)delta[b] = ((d12a[b] - sum_d )/ BS - (x_hat[b] * sum_xhat_d) / BS) / sqvar;
	*gamma += -eta * dg / BS;
	*beta += -eta * db / BS;
	



}
