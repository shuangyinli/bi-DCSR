//
// Created by shuangyinli on 2019-04-14.
//

#ifndef ESE_MAIN_H
#define ESE_MAIN_H
#include "stdio.h"
#include "util.h"
#include "train.h"
#include "pthread.h"

void begin_ese(char* settingfile, char* inputfile, char* model_root,char* beta_file = NULL, char* phi_file = NULL);


#endif //ESE_MAIN_H
