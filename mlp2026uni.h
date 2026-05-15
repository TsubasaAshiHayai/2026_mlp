#pragma once
#define NETMODEL 0  // 0;MLP 1 ;cnn
#define L 3
#define ETA 0.02 //0.001äwèKó¶
#define BATCHLEARN 1
#define BATCHNORM 1
#define BS 32
#define EPS 0.0000001
  
#define SIGMOID 0
#define SOFTCROSS 1 //soft and cross entropy
#define RELU 2



void InitW(int ns[], double woi[][NMAX][NMAX], double bj[][NMAX],double beta[][NMAX], double gamma[][NMAX]);
void Forward(int ns[], double out[][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX], int oact, int hact, double mmean[][NMAX], double mvar[][NMAX], double gamma[][NMAX], double beta[][NMAX]);
void BackProp(int ns[], double out[][NMAX], double Tk[], double delta[][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX],double eta, int oact, int hact);

void BForward(int ns[], double Bout[][BS][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX], int oact, int hact, double net[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX],double mmean[][NMAX],double mvar[][NMAX]);
void BBackProp(int ns[], double Bout[][BS][NMAX], double tk[BS][OD], double delta[][NMAX][BS], double woi[][NMAX][NMAX], double bj[][NMAX], double eta, int oact, int hact, double net[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX]);

void BatchNormF(double net[], double myu[], double* mean, double* var, double gamma, double beta);
void BatchNormB(double net[], double delta[], double mean, double var, double *gamma, double *beta, double eta);





