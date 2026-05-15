#define _USE_MATH_DEFINES
#include "DxLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "optidigit.h"
#include "mlp2026uni.h"


//*-------------------- MENU --------------------*//
#define MENUX 120				//メニューのX座標
#define MENUY 40				//メニューのY座標
#define BK 0					//黒
#define WH 16777215				//白
#define RD 16711680				//赤
#define GR 65280				//緑
#define BL 255					//青
//*-------------------- Setting --------------------*//
#define EMAX 30					// Epoch数
#define LOOP 1					// 試行回数
//*-------------------- Cosine Decay --------------------*//
#define CDECAY 0				// 0:通常 1:cosine decay
#define WARMUP 2				// ウォームアップ(Epoc数)
#define WARMUPRATE 0.2			// ウォームアップの係数
//*-------------------- Label Smoothing --------------------*//
#define LS 0					// 0:OFF, 1:ON
#define LSEPS 0.009				// パラメータ

int In, Tr, Epoc, TraOrder[TRA];
double Lrate, TrainAcc, TestAcc;
time_t End, Start;

double TraData[TRA][ID], TesData[TES][ID];
int TraClass[TRA], TesClass[TES];
char TraName[] = "optdigits_tra.csv", TesName[] = "optdigits_tes.csv";

double Out[L][NMAX], Tk[OD], Delta[L][NMAX];
double Woi[L - 1][NMAX][NMAX], Bj[L - 1][NMAX];
//int Ns[L] = { ID, NMAX, NMAX / 2, OD }; //L = 4
int Ns[L] = { ID, NMAX,OD }; //L = 3

//int Hact = SIGMOID,Oact = SIGMOID;
//int Hact = SIGMOID, Oact = SOFTCROSS;
int Hact = RELU, Oact = SOFTCROSS;
double BOut[L][BS][NMAX], BTk[BS][OD], BDelta[L][NMAX][BS];
double Net[L][NMAX][BS], Mean[L][NMAX], Var[L][NMAX], Beta[L][NMAX], Gamma[L][NMAX];
double MMean[L][NMAX], MVar[L][NMAX];
void Display();
void SetData(char traName[], char tesName[], double traData[][ID], double tesData[][ID],
	int traClass[], int tesClass[]);
void DeepLearn();

//int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_  HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd) {
	int MouseInput, MouseX, MouseY, j, ch, h, w;

	//srand((unsigned int)time(NULL));
	srand(1);
	SetData(TraName, TesName, TraData, TesData, TraClass, TesClass);
	InitW(Ns, Woi, Bj, Beta, Gamma);

	SetGraphMode(1200, 800, 32);	//画面モードの設定	
	SetBackgroundColor(255, 255, 255);
	ChangeWindowMode(TRUE);			//ウインドウモードに変更
	SetAlwaysRunFlag(TRUE);			//バックグラウンドでも実行を継続
	SetMouseDispFlag(TRUE);			//マウスを表示状態にする	
	if (DxLib_Init() == -1) return -1;	// ＤＸライブラリ初期化処理 エラーが起きたら直ちに終了

	Display();
	while (1) {
		if (ProcessMessage() == -1) break;	//エラーが起きたらループから抜ける
		MouseInput = GetMouseInput();	//マウスの入力を待つ			
		if ((MouseInput & MOUSE_INPUT_LEFT) != 0) {	//左ボタン押された
			GetMousePoint(&MouseX, &MouseY); //マウスの位置を取得
			if (MouseX < MENUX) { // Menu area click
				if (MouseY < MENUY) break;				//END
				else if (MouseY < MENUY * 2) 	InitW(Ns, Woi, Bj, Beta, Gamma);//NN初期化
				else if (MouseY < MENUY * 3) { //Input
					In = (rand() / (RAND_MAX + 1.0)) * TRA;
					//if (NETMODEL == 1) {//CNN
					//	for (j = 0, ch = 0; ch < ICH; ch++) for (h = 0; h < H; h++) for (w = 0; w < W; w++, j++)
					//		COut[0][0][ch][h][w] = TraData[In][j];
					//	CForward(ChN, Ks, COut, CWoi, CBias, Out[0], Cactv, CPower);
					//}
					//else 
					for (j = 0; j < ID; j++) Out[0][j] = TraData[In][j];//MLP
					Forward(Ns, Out, Woi, Bj, Hact, Oact, MMean, MVar, Gamma, Beta);
				}
				else if (MouseY < MENUY * 4)DeepLearn(); //Learn
				else if (MouseY < MENUY * 5) {
					for (Tr = 0; Tr < LOOP; Tr++) {
						InitW(Ns, Woi, Bj, Beta, Gamma);
						DeepLearn(); //Learn
					}
				}
			}
			Display();
		}
		WaitTimer(100);
	}
	DxLib_End();	//ＤＸライブラリ使用の終了処理
	return 0;		//ソフトの終了 
}

void Display() {
	int i, j, k, mx, my, cr, size = 64 / HD, b, max=0, s, c;
	char menu[][20] = { "END", "Init", "In Check", "DeepLearn", "N-Learn" };

	ClearDrawScreen();
	for (i = 0; i < 5; i++) {	//メニュー表示
		DrawString(10, i * MENUY + 10, menu[i], BK);
		DrawLine(0, MENUY * (i + 1), MENUX, MENUY * (i + 1), BK, 2);
	}
	DrawLine(MENUX, 0, MENUX, MENUY * 5, BK, 2);

	if (NETMODEL == 1) DrawString(10, MENUY * 6 + 10, "CNN", BK);
	else if (NETMODEL == 2) DrawString(10, MENUY * 6 + 10, "LinearCNN", BK);
	else DrawString(10, MENUY * 6 + 10, "MLP", BK);
	DrawFormatString(10, MENUY * 7 + 24, BK, "Learning Rate %.3lf", Lrate);
	DrawFormatString(10, MENUY * 8 + 10, BK, "Epoc %2d/%2d  Time %4d", Epoc, EMAX, End - Start);
	DrawFormatString(10, MENUY * 9, BK, "TraAcc %.3lf  %d", TrainAcc, TRA);
	DrawFormatString(10, MENUY * 9 + 24, BK, "TesAcc %.3lf  %d", TestAcc, TES);

	mx = MENUX + 100; my = MENUY;
	for (i = 0; i < HD; i++) for (j = 0; j < HD && (i * HD + j) < Ns[0]; j++) {//全結合層入力データの情報
		if (Out[0][i * HD + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
		else {
			cr = (int)(Out[0][i * HD + j] * 255);
			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
		}
	}
	my += size * HD;
	for (max = 0, k = 0; k < OD; k++) {// 全結合層 出力の表示
		if (Out[L - 1][max] < Out[L - 1][k]) max = k;
		DrawFormatString(mx, my + k * 20, BK, "%.2lf", Out[L - 1][k]);
	}
	DrawFormatString(mx, my + max * 20, RD, "%.2lf", Out[L - 1][max]);
	DrawFormatString(mx + 64, MENUY + 64, RD, "%d", max);

	//if (NETMODEL == 1 || NETMODEL == 4 || NETMODEL == 7) {//畳み込み層入力データの情報
	//	for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) { //入力
	//		if (COut[0][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//		else {
	//			if (ICH == 1) cr = GetColor((int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][0][i][j] * 255));
	//			else cr = GetColor((int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][1][i][j] * 255), (int)(COut[0][0][2][i][j] * 255));
	//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, cr, TRUE);
	//		}
	//	}
	//	for (my = 400, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1回畳み込み後入力データの情報
	//		if (COut[0][1][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//		else {
	//			cr = (int)(COut[0][1][0][i][j] * 255);
	//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//		}
	//	}
	//	for (my = 500, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
	//		if (COut[1][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//		else {
	//			cr = (int)(COut[1][0][0][i][j] * 255);
	//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//		}
	//	}
	//	//if (SEG > 2)
	//	//	for (my = 600, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
	//	//		if (COut[2][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//	//		else {
	//	//			cr = (int)(COut[2][0][0][i][j] * 255);
	//	//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//	//		}
	//	//	}
	//}
	//else if (NETMODEL == 2 || NETMODEL == 5) {//畳み込み層入力データの情報
	//	for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) { //入力
	//		if (LCOut[0][0][i * W + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//		else {
	//			if (ICH == 1) cr = GetColor((int)(LCOut[0][0][i * W + j] * 255), (int)(LCOut[0][0][i * W + j] * 255), (int)(LCOut[0][0][i * W + j] * 255));
	//			else cr = GetColor((int)(LCOut[0][0][i * W + j] * 255), (int)(LCOut[0][0][H * W + i * W + j] * 255), (int)(LCOut[0][0][2 * H * W + i * W + j] * 255));
	//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, cr, TRUE);
	//		}
	//	}
	//	for (my = 400, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1回畳み込み後入力データの情報
	//		if (LCOut[0][1][i * W + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//		else {
	//			cr = (int)(LCOut[0][1][i * W + j] * 255);
	//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//		}
	//	}
	//	for (my = 500, i = 0; i < RD / 2; i++) for (j = 0; j < RD / 2; j++) {//プーリング後入力データの情報
	//		if (LCOut[1][0][i * W / 2 + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//		else {
	//			cr = (int)(LCOut[1][0][i * W / 2 + j] * 255);
	//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//		}
	//	}
	//	//if (SEG > 2)
	//	//	for (my = 600, i = 0; i < RD/4; i++) for (j = 0; j < RD/4; j++) {//プーリング後入力データの情報
	//	//		if (LCOut[2][0][i * W/4 + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//	//		else {
	//	//			cr = (int)(LCOut[2][0][i * W/4 + j] * 255);
	//	//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//	//		}
	//	//	}
	//}

	//if (BATCHLEARN) {
	//	DrawFormatString(10, MENUY * 7, BK, "BatchLearn %d", BS);
	//	for (mx = MENUX + 200, b = 0; b < BS && b < 10; b++, mx += 80) {
	//		for (my = MENUY, i = 0; i < RD; i++) for (j = 0; j < RD && (i * RD + j) < Ns[0]; j++) {//全結合層入力データの情報
	//			if (BOut[0][b][i * RD + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//			else {
	//				cr = (int)(BOut[0][b][i * RD + j] * 255);
	//				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//			}
	//		}
	//		my = MENUY + size * RD;
	//		for (max = 0, k = 0; k < OD; k++) {// 全結合層 出力の表示
	//			if (BOut[L - 1][b][max] < BOut[L - 1][b][k]) max = k;
	//			DrawFormatString(mx, my + k * 20, BK, "%.2lf", BOut[L - 1][b][k]);
	//		}
	//		DrawFormatString(mx, my + max * 20, RED, "%.2lf", BOut[L - 1][b][max]);
	//		if (BTk[b][max] > 0.5) DrawFormatString(mx + 64, MENUY + 64, RED, "%d", max);
	//		else DrawFormatString(mx + 64, MENUY + 64, BLU, "%d", max);

	//		if (NETMODEL == 1) {//畳み込み層入力データの情報
	//			for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {
	//				if (BCOut[0][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//				else {
	//					if (ICH == 1) cr = GetColor((int)(BCOut[0][0][0][b][i][j] * 255), (int)(BCOut[0][0][0][b][i][j] * 255), (int)(BCOut[0][0][0][b][i][j] * 255));
	//					else cr = GetColor((int)(BCOut[0][0][0][b][i][j] * 255), (int)(BCOut[0][0][1][b][i][j] * 255), (int)(BCOut[0][0][2][b][i][j] * 255));
	//					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, cr, TRUE);
	//				}
	//			}
	//			for (my = 400, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1回畳み込み後入力データの情報
	//				if (BCOut[0][1][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//				else {
	//					cr = (int)(BCOut[0][1][0][b][i][j] * 255);
	//					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//				}
	//			}
	//			for (my = 500, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
	//				if (BCOut[1][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//				else {
	//					cr = (int)(BCOut[1][0][0][b][i][j] * 255);
	//					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//				}
	//			}
	//			//if (SEG > 2)
	//			//	for (my = 600, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
	//			//		if (BCOut[2][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//			//		else {
	//			//			cr = (int)(BCOut[2][0][0][b][i][j] * 255);
	//			//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//			//		}
	//			//	}
	//		}
	//		else if (NETMODEL == 2) {//畳み込み層入力データの情報
	//			for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {
	//				if (BLCOut[0][0][b][i * W + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//				else {
	//					if (ICH == 1) cr = GetColor((int)(BLCOut[0][0][b][i * W + j] * 255), (int)(BLCOut[0][0][b][i * W + j] * 255), (int)(BLCOut[0][0][b][i * W + j] * 255));
	//					else cr = GetColor((int)(BLCOut[0][0][b][i * W + j] * 255), (int)(BLCOut[0][0][b][H * W + i * W + j] * 255), (int)(BLCOut[0][0][b][2 * H * W + i * W + j] * 255));
	//					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, cr, TRUE);
	//				}
	//			}
	//			for (my = 400, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1回畳み込み後入力データの情報
	//				if (BLCOut[0][1][b][i * W + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//				else {
	//					cr = (int)(BLCOut[0][1][b][i * W + j] * 255);
	//					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//				}
	//			}
	//			for (my = 500, i = 0; i < RD / 2; i++) for (j = 0; j < RD / 2; j++) {//プーリング後入力データの情報
	//				if (BLCOut[0][1][b][i * W / 2 + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//				else {
	//					cr = (int)(BLCOut[0][1][b][i * W / 2 + j] * 255);
	//					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//				}
	//			}
	//			//if (SEG > 2)
	//			//	for (my = 600, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
	//			//		if (BCOut[2][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
	//			//		else {
	//			//			cr = (int)(BCOut[2][0][0][b][i][j] * 255);
	//			//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
	//			//		}
	//			//	}
	//		}
	//	}
	//}
	ScreenFlip(); WaitTimer(10);
}

void SetData(char traName[], char tesName[], double traData[][ID], double tesData[][ID],
	int traClass[], int tesClass[]) {
	FILE* fin;
	int i, j, temp;

	fopen_s(&fin, traName, "r");
	for (i = 0; i < TRA; i++) {
		for (j = 0; j < ID; j++) {
			fscanf_s(fin, "%d,", &temp);
			traData[i][j] = (double)temp / VMAX;
		}
		fscanf_s(fin, "%d\n", &traClass[i]);
	}
	fclose(fin);

	fopen_s(&fin, tesName, "r");
	for (i = 0; i < TES; i++) {
		for (j = 0; j < ID; j++) {
			fscanf_s(fin, "%d,", &temp);
			tesData[i][j] = (double)temp / VMAX;
		}
		fscanf_s(fin, "%d\n", &tesClass[i]);
	}
	fclose(fin);
}

void DeepLearn() {
	int t, i, j, temp, count, b, ch, h, w, max;
	double ceta, loss;

	FILE* fp;

	if (BATCHLEARN) {
		if (NETMODEL == 1) fopen_s(&fp, "CNN_Batch.csv", "a");
		else fopen_s(&fp, "MLP_Batch.csv", "a");
	}
	else {
		if (NETMODEL == 1) fopen_s(&fp, "CNN_log.csv", "a");
		else fopen_s(&fp, "MLP_log.csv", "a");
	}
	if (fp == NULL) return;
	else fprintf_s(fp, "\nEpoc,TrainAcc,TestAcc,Time,TraLoss,TesLoss\n");

	Start = time(NULL);	End = time(NULL);
	for (Epoc = 0; Epoc < EMAX; Epoc++) { // Train data 一通り学習
		for (t = 0; t < TRA; t++) TraOrder[t] = t;
		for (t = 0; t < TRA - 1; t++) {
			i = rand() / (RAND_MAX + 1.0) * (TRA - (double)t);
			temp = TraOrder[t];
			TraOrder[t] = TraOrder[t + i];
			TraOrder[t + i] = temp;
		}
		Lrate = ETA; // cosine decay 学習率を可変にする制御

		Display();
		
		if (BATCHLEARN) {
			Lrate = ETA * BS / 2.0;
			for (count = 0, t = 0; t <= TRA - BS; t += BS) {
				for (b = 0; b < BS; b++) {
					//in = (rand() / (RAND_MAX + 1.0)) * TRA;
					i = TraOrder[t + b];
					/*if (NETMODEL == 1)for (j = 0, ch = 0; ch < ICH; ch++) for (h = 0; h < H; h++) for (w = 0; w < W; w++, j++)BCOut[0][0][ch][b][h][w] = TraData[i][j];*/
					
					//else 
					for (j = 0; j < ID; j++) BOut[0][b][j] = TraData[i][j];
					/* 教師データ作成 */
					//if (LS) {// ラベル平滑化
					//	for (j = 0; j < OD; j++) BTk[b][j] = lsk;
					//	BTk[b][TraClass[i]] = 1.0 - LSEPS;// hot-one vector
					//}
					//else {
						for (j = 0; j < OD; j++) BTk[b][j] = 0.0;
						BTk[b][TraClass[i]] = 1.0;// hot-one vector
					//}
				}
				/*if (NETMODEL == 1) {
					BCForward(ChN, Ks, BCOut, CWoi, CBias, BOut[0]);
					BForward(Ns, BOut, Woi, Bias, RELU, SOFTMAX, BNet, Mean, Var, MMean, MVar, Gamma, Beta);
					BBackProp(Lrate, Ns, BOut, Woi, Bias, BDelta, BTk, RELU, SOFTMAX, BNet, Mean, Var, Gamma, Beta);
					BCBackProp(Lrate, ChN, Ks, BCOut, CWoi, CBias, BDelta[0], BCDelta, DCWoi, DCBias);
				}
			
				else {*/
					BForward(Ns, BOut, Woi, Bj, Oact, Hact,Net,Mean, Var, Gamma, Beta, MMean, MVar);
					BBackProp(Ns, BOut, BTk, BDelta, Woi, Bj, Lrate, Oact, Hact, Net, Mean, Var, Gamma, Beta);
				//}

				for (b = 0; b < BS; b++) {
					for (i = 0, j = 1; j < OD; j++) if (BOut[L - 1][b][i] < BOut[L - 1][b][j]) i = j;
					if (BTk[b][i] > 0.5)count++;
				}
			}
			TrainAcc = (double)count / (double)t;
			
		}
		else {
			for (count = 0, t = 0; t < TRA; t++) {
				In = TraOrder[t];
				for (j = 0; j < OD; j++) Tk[j] = 0.0;
				Tk[TraClass[In]] = 1.0;// hot-one vector

				//if (NETMODEL == 1) {//CNN
				//	for (j = 0, ch = 0; ch < ICH; ch++) for (h = 0; h < H; h++) for (w = 0; w < W; w++, j++)
				//		COut[0][0][ch][h][w] = TraData[In][j];
				//	CForward(ChN, Ks, COut, CWoi, CBias, Out[0], Cactv, CPower);
				//	Forward(Ns, Out, Woi, Bias, RELU, SOFTMAX, MMean, MVar, Gamma, Beta, TRAIN);
				//	BackProp(Lrate, Ns, Out, Woi, Bias, Delta, Tk, RELU, SOFTMAX, NETMODEL);
				//	CBackProp(Lrate, ChN, Ks, COut, CWoi, CBias, Delta[0], CDelta, DCWoi, DCBias, Cactv, CPower);
				//}
				//else {//MLP
				for (j = 0; j < ID; j++) Out[0][j] = TraData[In][j];
				Forward(Ns, Out, Woi, Bj, Oact, Hact, MMean, MVar, Gamma, Beta);
				BackProp(Ns, Out, Tk, Delta, Woi, Bj, Lrate, Oact, Hact);


				for (max = 0, i = 1; i < OD; i++) {
					if (Out[L - 1][max] < Out[L - 1][i]) max = i;
				}
				if (max == TraClass[In]) count++;

			}
			TrainAcc = (double)count / (double)t;
		}
			for (count = 0, t = 0; t < TES; t++) { // 評価時
				//if (NETMODEL == 1) {//CNN
				//	for (j = 0, ch = 0; ch < ICH; ch++) for (h = 0; h < H; h++) for (w = 0; w < W; w++, j++)
				//		COut[0][0][ch][h][w] = TesData[t][j];
				//	CForward(ChN, Ks, COut, CWoi, CBias, Out[0], Cactv, CPower);
				//	Forward(Ns, Out, Woi, Bias, RELU, SOFTMAX, MMean, MVar, Gamma, Beta, TEST);
				//}
				//else {//MLP
				for (j = 0; j < ID; j++) Out[0][j] = TesData[t][j];
				Forward(Ns, Out, Woi, Bj,Oact,Hact, MMean, MVar, Gamma, Beta);
				
				
				//}

				for (i = 0, j = 1; j < OD; j++) {
					if (Out[L - 1][i] < Out[L - 1][j]) i = j;
				}
				if (i == TesClass[t]) count++;
			}
			TestAcc = (double)count / TES;
			End = time(NULL);
			fprintf_s(fp, "%d,%.4lf,%.4lf,%d\n", Epoc, TrainAcc, TestAcc, (int)(End - Start));
		
		Display();
	}
	fclose(fp);
}
