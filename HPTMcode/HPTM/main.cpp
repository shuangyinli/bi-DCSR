#include <iostream>
#include "main.h"
#include "data.h"

void begin_ese(char* settingfile, char* inputfile, char* model_root,char* beta_file,
               char* phi_file) {
    setbuf(stdout, NULL);
    printf("Read the configuration... \n");
    Configuration config = Configuration(settingfile);
    int neighbor = config.neighbor;

    int num_topics = config.num_topics;

    int num_docs;
    int num_words;
    int num_all_words;
    int num_keywords;

    srand(unsigned(time(0)));
    printf("Read the training data... \n");
    Document** corpus = readData(inputfile,num_topics,num_words, num_docs,
                                 num_all_words, num_keywords, neighbor);

    printf("Build the Model... \n");
    Model * model = new Model(num_docs,num_topics,num_words,num_keywords,neighbor);

    time_t learn_begin_time = time(0);
    int num_round = 0;

    // init the beta and topics

    if(beta_file){
        readinitbeta(model, beta_file);
    }

    if(phi_file){
        readinitphi(model, phi_file);
    }

    printf("The training set contains %d documents, and the number of words in dictionary is %d.\n", model->num_docs, model->num_words);
    printf("The number of keywords is %d. \n", model->num_all_keywords);
    printf("%d neighborhood sentences are considered. \n", model->neighbor);

    float plik =0.0;
    float lik=0.0;
    float converged = 1;
    puts("Now begin to train: ");
    do {
        time_t cur_round_begin_time = time(0);
        plik = lik;
        printf("Round %d -> inference... ", num_round);
        runThreadInference(corpus, model, &config, num_docs);
        printf("pi... ");
        learnPi(corpus, model, &config);

        run_thread_learn_phi_beta(corpus, model, &config);

        lik = 0.0;
        for(int d = 0; d < num_docs; d++){
            lik += corpus[d]->doclik;
        }
        float perplexity = exp(-lik/num_all_words);

        converged = (plik - lik) / plik;
        if (converged < 0) config.sen_max_var_iter *= 2;

        unsigned int cur_round_cost_time = time(0) - cur_round_begin_time;

        printf("loglikelihood =%lf, perplexity=%lf, converged=%lf, time=%u secs.\n",
               lik, perplexity, converged, cur_round_cost_time);

        num_round += 1;

        if (num_round % 10 == 0){
            printParameters(corpus,num_round, model_root, model);
        }
    }
    while (num_round < config.max_em_iter && (converged < 0 || converged > config.em_converence || num_round < 10));
    unsigned int learn_cost_time = time(0) - learn_begin_time;
    printf("all learn runs %d rounds and cost %u secs.\n", num_round, learn_cost_time);
    printParameters(corpus,-1,model_root, model);

    delete model;
    for (int i = 0; i < num_docs; i++) delete corpus[i];
    delete[] corpus;
}

void inference_ese(char* settingfile, char* testfile,char* pi_file, char* beta_file,
               char* phi_file, char* model_root){
    setbuf(stdout, NULL);
    printf("Read the configuration... \n");
    Configuration config = Configuration(settingfile);
    int neighbor = config.neighbor;

    int num_topics = config.num_topics;
    int num_words = config.num_words;
    int num_all_keywords = config.num_all_keywords;

    int num_test_docs;
    int num_test_words;
    int num_all_test_words;
    int num_test_keywords;

    srand(unsigned(time(0)));
    printf("Read the training data... \n");
    Document** corpus = readData(testfile,num_topics,num_test_words, num_test_docs,
                                 num_all_test_words, num_test_keywords, neighbor);

    printf("Build the Model... \n");

    Model * model = new Model(pi_file,phi_file, beta_file, num_topics,neighbor, num_words, num_all_keywords, num_test_docs);

    time_t learn_begin_time = time(0);
    int num_round = 0;

    printf("The inference set contains %d documents, and the number of words in dictionary is %d.\n", num_test_docs, model->num_words);
    printf("The number of keywords is %d. \n", model->num_all_keywords);
    printf("%d neighborhood sentences are considered. \n", model->neighbor);

    float plik =0.0;
    float lik=0.0;
    float converged = 1;
    puts("Now begin to inference: ");
    do {
        time_t cur_round_begin_time = time(0);
        plik = lik;
        printf("Round %d -> inference... ", num_round);
        runThreadInference(corpus, model, &config, num_test_docs);

        lik = 0.0;
        for(int d = 0; d < num_test_docs; d++) lik += corpus[d]->doclik;

        float perplexity = exp(-lik/num_all_test_words);

        converged = (plik - lik) / plik;
        if (converged < 0) num_round = config.max_em_iter;

        unsigned int cur_round_cost_time = time(0) - cur_round_begin_time;

        printf("loglikelihood =%lf, perplexity=%lf, converged=%lf, time=%u secs.\n", lik, perplexity, converged, cur_round_cost_time);

        num_round += 1;
        printtestresults(corpus,num_round, model_root, model);

    }
    while (num_round < config.max_em_iter && (converged < 0 || converged > config.em_converence || num_round < 10));
    unsigned int learn_cost_time = time(0) - learn_begin_time;
    printf("all learn runs %d rounds and cost %u secs.\n", num_round, learn_cost_time);
    //printtestresults(corpus,-1,model_root, model);

    delete model;
    for (int i = 0; i < num_test_docs; i++) delete corpus[i];
    delete[] corpus;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && argc == 5 && strcmp(argv[1],"est") == 0) {
        printf("\n");
        begin_ese(argv[2],argv[3], argv[4], NULL, NULL);
    }else if(argc > 1 && argc == 6 && strcmp(argv[1],"est") == 0){
        printf("Now begin training with initial beta... \n");
        begin_ese(argv[2],argv[3], argv[4], argv[5],NULL);
    }
    else if(argc > 1 && argc == 7 && strcmp(argv[1],"est") == 0){
        printf("Now begin training with initial beta and phi... \n");
        begin_ese(argv[2],argv[3], argv[4], argv[5],argv[6]);
    } else if(argc > 1 && argc == 8 && strcmp(argv[1],"inf") == 0){
        printf("Now begin inference with pi,beta and phi... \n");
        inference_ese(argv[2],argv[3], argv[4], argv[5],argv[6],argv[7]);

    }else {
        printf("Please use the following setting.\n");
        printf("\n");
        printf("************* Trainning ***********************\n");

        printf(
                "./ese est <setting.txt>  <input data file> <model save dir>\n\n");
        printf(
                "If you want to initialize the model with default parameters, please use: \n");
        printf(
                "./ese est <setting.txt> <input data file> <model save dir> <beta file > <phi file>\n\n");
        printf("**************Inference**********************\n");
        printf(
                "./ese inf <setting.txt> <input test data file> <pi> <beta> <phi> <output dir>\n");
        printf("\n");
    }
    return 0;
}