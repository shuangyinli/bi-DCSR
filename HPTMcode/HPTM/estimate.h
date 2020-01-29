//
// Created by shuangyinli on 2019-04-14.
//

#ifndef ESE_ESTIMATE_H
#define ESE_ESTIMATE_H

#include "data.h"
#include "math.h"
#include "util.h"
#include "vector"
#include "set"
#include "algorithm"
#include "ctime"
#include "cstdlib"
#include "random"
#include "chrono"
using namespace std;

struct Thread_Data_phi_beta {
    Document ** corpus;
    int start;
    int end;
    Model* thread_model;
    Thread_Data_phi_beta(Document** corpus_, int start_, int end_, Model* thread_model_) : corpus(corpus_), start(start_), end(end_), thread_model(thread_model_) {
    }
};

struct Thread_Data_phi {
    Document ** corpus;
    int start;
    int end;
    Model* thread_model;
    Thread_Data_phi(Document** corpus_, int start_, int end_, Model* thread_model_) : corpus(corpus_), start(start_), end(end_), thread_model(thread_model_) {
    }
};


float corpuslikelihood(Document** corpus, Model* model);
float compute_document_likelihood(Document* doc, Model* model);
float compute_sentence_likehood(Sentence* sentence, Model* model);
float getPiFunction(Document** corpus, Model* model, int num_docs);
void getDescentPi(Document** corpus, Model* model, float* descent_pi, int num_docs);
void learnPi(Document** corpus, Model* model, Configuration* configuration);
void* thread_learn_phi_beta(void* thread_data_theta_beta);
void run_thread_learn_phi_beta(Document** corpus, Model* model, Configuration* configuration);
void normalize_matrix_rows(float* mat, int rows, int cols);
void normalize_log_matrix_rows(float* log_mat, int rows, int cols);
void* thread_learn_phi(void* thread_data_theta);
void run_thread_learn_phi(Document** corpus, Model* model, Configuration* configuration);


#endif //ESE_ESTIMATE_H
