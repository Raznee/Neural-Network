#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*---<USER DEFINES>---*/

#define INPUT_NUMBER 2
#define OUTPUT_NUMBER 2
#define HIDDEN_NUMBER 2

//#define ACT Sigmoid
//#define DERACT derSigmoid

#define ACT ReLU
#define DERACT derReLU

#define LEARNING_RATE 1 //(0.5)
#define L_RELU_SCALING_VALUE 1 //(0.01)

/*---</USER DEFINES>---*/

#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define MIN(a,b) (((a)<(b)) ? (a) : (b))

#ifndef HIDDEN_NUMBER
#define HIDDEN_NUMBER (unsigned int)MIN(MIN(((InputNumber + OutputNumber) / 2), (2 * InputNumber - 1)), ((InputNumber + OutputNumber) * 2 / 3))
#endif

double vec[] = { 1, 3.14, -2, 7, -100, -9, 0, -5.43, 4.1, -19.73 };
double v[] = { -6, -4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 6 };

const unsigned int InputNumber = INPUT_NUMBER;
const unsigned int OutputNumber = OUTPUT_NUMBER;
const unsigned int HiddenNumber = HIDDEN_NUMBER;
/*const unsignedint NumPattern = 100;*/

double Input[InputNumber];
double Hidden[HiddenNumber];
double Output[OutputNumber];

double Target[OutputNumber];

double C[OutputNumber];

double Weight_I_H[InputNumber][HiddenNumber];
double Weight_H_O[HiddenNumber][OutputNumber];

/*Temp weights*/
double tWIH[InputNumber][HiddenNumber];
double tWHO[HiddenNumber][OutputNumber];

double *p1 = &Weight_I_H[0][0];
double *p2 = &Weight_H_O[0][0];

double Bias_H[HiddenNumber];
double Bias_O[OutputNumber];

/*Temp biases*/
double tBH[HiddenNumber];
double tBO[OutputNumber];

double aZO[OutputNumber];
double aZH[HiddenNumber];
double aAO[OutputNumber];
double aAH[HiddenNumber];

double Error;

/*---FUNCTIONS AND THEIR DERIVATIVES---*/

double
ReLU (double input)
{
  if (input <= 0)
    return 0;
  else
    return input;
}

double
derReLU (double input)
{
  double result;
  if (input <= 0)
    result = 0;
  else
    result = 1;
  return result;
}

double
leakyReLU (double input)
{
  double result;
  if (input <= 0)
    result = input * L_RELU_SCALING_VALUE;
  else
    result = input;
  return result;
}

double
derLeakyReLU (double input)
{
  double result;
  if (input <= 0)
    result = L_RELU_SCALING_VALUE;
  else
    result = 1;
  return result;
}

double
Sigmoid (double input)
{
  return 1 / (1 + exp ((-1) * input));
}

double
derSigmoid (double input)
{
  return (Sigmoid (input) * (1 - Sigmoid (input)));
}

double
derTanh (double input)
{
  return (1 - (tanh (input) * tanh (input)));
}


double
Activation (double d)
{
  double (*f) (double) = ACT;
   return (*f) (d);
}

double
derActivation (double d)
{
  double (*f) (double) = DERACT;
  (*f) (d);
}

double
Random (void)
{
  int k;
  double r;
  k = rand () % 100;
  r = (double) k;
  r /= 100.00;
  r += 0.005;
  return r;
}

/*
void
Calculate (void)
{
  int i, j, k;			//int p; 
  double Temp = 0.0;
  //Error = 0.0;
  //for (p = 1; p <= NumPattern; p++) {
  for (j = 0; j < HiddenNumber; j++)
    {
      for (i = 0; i < InputNumber; i++)
	Temp += Input[i] * Weight_I_H[i][j];
      Hidden[j] = Sigmoid (Bias_H[j] + Temp);
    }
  for (j = 0; j < OutputNumber; j++)
    {
      for (i = 0; i < HiddenNumber; i++)
	Temp += Hidden[i] * Weight_H_O[i][j];
      Output[j] = Sigmoid (Bias_O[j] + Temp);
    }


  //}
}*/

/*
I:  input
W:  weight
B:  bias
T:  target
Z:  Z=N#(W*I)+B
A:  A=a(Z)
C:  C=N#(T-Z)^2
*/

/*---F---*/

double
ZH (unsigned sn)
{
  unsigned i;
  double temp = Bias_H[sn];
  for (i = 0; i < HiddenNumber; i++)
    {
      temp += (Weight_I_H[i][sn]) * (Input[i]);
    }
  // printf ("ZH[%d]: %f\n", sn, temp);
  return temp;

}

double
ZO (unsigned sn)
{
  unsigned i;
  double temp = Bias_O[sn];
  for (i = 0; i < OutputNumber; i++)
    {
      temp += (Weight_H_O[i][sn]) * (Hidden[i]);
    }
  //printf ("ZO[%d]: %f\n", sn, temp);
  return temp;
}

void
StoreZO (void)
{
  int i = 0;
  for (i = 0; i < OutputNumber; i++)
    aZO[i] = ZO (i);
  //temp_O[0][i] = ZO (i);
}

void
StoreAO (void)
{
  double d;
  int i = 0;
  for (i = 0; i < OutputNumber; i++)
    {
      d = Activation (aZO[i]);
      //printf ("AO[%d]: %f\n", i, d);
      aAO[i] = d;
    }
}

void
StoreZH (void)
{
  int i;
  for (i = 0; i < HiddenNumber; i++)
    {				//printf("???: %f\n", ZH (i));
      aZH[i] = ZH (i);
    }
}

void
StoreAH (void)
{
  double d;
  int i = 0;
  for (i = 0; i < HiddenNumber; i++)
    {
      d = Activation (aZH[i]);
      //printf ("AH[%d]: %f\n", i, d);
      aAH[i] = d;
    }
}



/*---DERIVATE---*/

//O->H
// Total Cost / AO[n]
double
dCO_per_dAO (unsigned i)
{
  double d;
  d = -(Target[i] - aAO[i]);
  //printf ("Total Cost / AO[%d]: %f\n", i, d);
  return d;

}

//O->H
// AO[n]  / ZO[n]
double
dAO_per_dZO (unsigned n)
{
  double d = derActivation (aZO[n]);
  //printf ("AO[%d]  / ZO[%d]: %f\n", n, n, d);
  return d;
}

//H->I
// AH[n]  / ZH[n]
double
dAH_per_dZH (unsigned n)
{
  double d = derActivation (aZH[n]);
  //printf ("AH[%d]  / ZH[%d]: %f\n", n, n, d);
  return d;
}

//O->H
// ZO[n]  /  W[k]
double
dZO_per_dW (unsigned n)
{
  //printf ("ZO[%d]  /  W[k]: %f\n", n, aAH[n]);
  return aAH[n];
}

//H->I
// ZH[n]  /  W[k]
double
dZH_per_dW (unsigned n)
{
  //printf ("ZH[%d]  /  W[k]: %f\n", n, Input[n]);
  return Input[n];
}

//O->H
// ZO[n]  /  AH[k]
double
dZO_per_dAH (unsigned i, unsigned j)
{
  //printf ("ZO[%d]  /  AH[%d]: %f\n", i, j, Weight_H_O[i][j]);
  return Weight_H_O[i][j];
}

//H->I
// dZH[j] / dI[i]  
double
dZH_per_dI (unsigned i, unsigned j)
{
  //printf ("dZH[%d] / dI[%d]: %f\n", j, i, Weight_I_H[i][j]);
  return Weight_I_H[i][j];
}

//H->I
// Total Cost / AO[n]
double
dCH_per_dAH (unsigned n)
{
  double d;
  d = dCO_per_dAO (n) * dAO_per_dZO (n);
  //printf ("Total Cost / AO[%d]: %f\n", n, d);
  return d;
}

/*----------------------------------------------------------------------*/

double
Cost (unsigned i)
{
  return (Output[i] - Target[i]) * (Output[i] - Target[i]);
}

void
StoreCost (void)
{
  int i;
  for (i = 0; i < OutputNumber; i++)
    C[i] = Cost (i);
}

double
CostSum (void)
{				//double TempVector[OutputNumber];
  int i = 0;
  double Sum = 0;
  for (i = 0; i < OutputNumber; i++)
    Sum += C[i];
  return Sum / OutputNumber;
}

void
UpdateH (void)
{
  int i;
  for (i = 0; i < HiddenNumber; i++)
    Hidden[i] = aAH[i];
}

void
UpdateO (void)
{
  int i;
  for (i = 0; i < OutputNumber; i++)
    Output[i] = aAO[i];
}

void
UpdateWHO (void)
{
  int j, k;
  for (j = 0; j < HiddenNumber; j++)
    for (k = 0; k < OutputNumber; k++)
      Weight_H_O[j][k] = tWHO[j][k];
}

void
UpdateWIH (void)
{
  int i, j;
  for (i = 0; i < InputNumber; i++)
    for (j = 0; j < HiddenNumber; j++)
      Weight_I_H[i][j] = tWIH[i][j];
}

void
FeedForward (void)
{
  StoreZH ();
  StoreAH ();
  UpdateH ();
  StoreZO ();
  StoreAO ();
  UpdateO ();
  StoreCost ();
}

void
//double
Backpropagate (void)
{
  unsigned i, j, k;
  double tt = 0, temp = 0;

  //Weight_H_O
  for (i = 0; i < HiddenNumber; i++)
    {
      for (j = 0; j < OutputNumber; j++)
	{
	  temp = dCO_per_dAO (j) * dAO_per_dZO (j) * dZO_per_dW (j);
	  tWHO[i][j] = Weight_H_O[i][j] - (LEARNING_RATE * temp);
	  printf ("W%d: %f\n", i * 2 + j + 4, tWHO[i][j]);
	}
    }
    printf ("\n");

  //Weight_I_H
  for (i = 0; i < InputNumber; i++)
    {

      for (j = 0; j < HiddenNumber; j++)
	{
	  temp = 0;
	  for (k = 0; k < OutputNumber; k++)
	    {
	      temp +=
		dZO_per_dAH (j,
			     k) * (-(Target[k] - aAO[k])) * dAO_per_dZO (k);
	    }
	  tt = temp * dAH_per_dZH (i) * dZH_per_dW (i);

	  tWIH[i][j] = temp = dZH_per_dI (i, j) - LEARNING_RATE * tt;
	  printf ("W%d: %f\n", i * 2 + j, tWIH[i][j]);
	}
    }
    printf ("\n");
    
    //Bias_H
    for (i = 0; i < InputNumber; i++)
    {

      for (j = 0; j < HiddenNumber; j++)
	{
	  temp = 0;
	  for (k = 0; k < OutputNumber; k++)
	    {
	      temp +=
		dZO_per_dAH (j,
			     k) * (-(Target[k] - aAO[k])) * dAO_per_dZO (k);
	    }
	  tt = temp * dAH_per_dZH (i);
        printf("B%d: %f\n",i * 2 + j, tt);
	  tWIH[i][j] = temp = dZH_per_dI (i, j) - LEARNING_RATE * tt;
	  //printf ("W%d: %f\n", i * 2 + j, tWIH[i][j]);

	}
    }
    printf ("\n");
    

}

void
Init_Random (void)
{
  int i, j, k;

  for (j = 0; j < HiddenNumber; j++)
    {
      for (i = 0; i < InputNumber; i++)
	Weight_I_H[i][j] = Random ();

      Bias_H[j] = Random ();
    }
  for (k = 0; k < OutputNumber; k++)
    {
      for (j = 0; j < HiddenNumber; j++)
	Weight_H_O[j][k] = Random ();

      Bias_O[k] = Random ();
    }
}

void
Init_Example (void)
{
  Input[0] = 0.05;
  Input[1] = 0.1;

  Target[0] = 0.01;
  Target[1] = 0.99;

  Weight_I_H[0][0] = 0.15;
  Weight_I_H[0][1] = 0.25;
  Weight_I_H[1][0] = 0.20;
  Weight_I_H[1][1] = 0.30;

  Weight_H_O[0][0] = 0.40;
  Weight_H_O[0][1] = 0.50;
  Weight_H_O[1][0] = 0.45;
  Weight_H_O[1][1] = 0.55;

  Bias_H[0] = 0.35;
  Bias_H[1] = 0.35;

  Bias_O[0] = 0.60;
  Bias_O[1] = 0.60;
}

int
main ()
{
    double CS;
    
  //Init
  Init_Example ();

  //Cycle
  FeedForward ();
  CS=CostSum ();
  printf ("CostSum: %f\n", CS);
  Backpropagate ();
  UpdateWHO ();
  UpdateWHO ();
  FeedForward ();
  CS=CostSum ();
  printf ("CostSum: %f\n", CS);
  return 0;
}


