//
// Created by shuangyinli on 2019-04-14.
//

#ifndef ESE_TRAIN_H
#define ESE_TRAIN_H

#include "data.h"
#include "estimate.h"
#include "math.h"
#include "util.h"

struct ThreadData {
    Document** corpus;
    int start;
    int end;
    Configuration* configuration;
    Model* model;
    ThreadData(Document** corpus_, int start_, int end_, Configuration* configuration_, Model* model_) : corpus(corpus_), start(start_), end(end_), configuration(configuration_), model(model_) {
    }
};

void doInference(Document* doc, Model* model, Configuration* configuration);
void computeDocTopicDistribution(Document* doc, Model* model);
void inferenceGamma(Sentence* sentence, Model* model);
void inferenceXi(Sentence* sentence, Model* model,Configuration* configuration);
void getDescentXi(Sentence* sentence, Model* model,float* descent_xi);
float getXiFunction(Sentence* sentence, Model* model);
inline void initXi(float* xi,int num);

void* ThreadInference(void* thread_data);
void runThreadInference(Document** corpus, Model* model, Configuration* configuration, int num_docs);


#endif //ESE_TRAIN_H
