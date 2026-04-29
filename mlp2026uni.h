#pragma once
#define NETMODEL 0  // 0;MLP 1 ;cnn
#define L 3
#define ETA 0.02 //0.001äwèKó¶
#define BATCHLEARN 0
#define BS 32
 
#define SIGMOID 0
#define SOFTCROSS 1 //soft and cross entropy
#define RELU 2



void InitW(int ns[], double woi[][NMAX][NMAX], double bj[][NMAX]);
void Forward(int ns[], double out[][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX], int oact, int hact);
void BackProp(int ns[], double out[][NMAX], double Tk[], double delta[][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX],double eta, int oact, int hact);

void BForward(int ns[], double Bout[][BS][NMAX], double woi[][NMAX][NMAX], double bj[][NMAX], int oact, int hact);
void BBackProp(int ns[], double Bout[][BS][NMAX], double tk[BS][OD], double delta[][NMAX][BS], double woi[][NMAX][NMAX], double bj[][NMAX], double eta, int oact, int hact);








