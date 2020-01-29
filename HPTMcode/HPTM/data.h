//
// Created by shuangyinli on 2019-04-14.
//

#ifndef ESE_DATA_H
#define ESE_DATA_H

#include "stdio.h"
#include "fstream"
#include "iostream"

using namespace std;

struct Sentence{
    float* xi;
    float* log_gamma;
    int* words_ptr;
    int* words_cnt_ptr;
    int* keyword_ptr;
    int num_words;
    int num_topics;
    float* log_topic;
    float senlik; //log
    int neighbor;
    int num_keyword;
    float* log_neighbortopics; //the neighbor sentences
    Sentence(int* words_ptr_, int* words_cnt_ptr_, int * keyword_ptr_, int num_words_, int num_keyword_, int num_topics_, int neighbor_){
        words_ptr = words_ptr_;
        words_cnt_ptr = words_cnt_ptr_;
        num_words = num_words_;
        num_topics = num_topics_;
        neighbor = neighbor_;
        xi = new float[2*neighbor_ + num_keyword_];
        log_gamma= new float[num_words_ * num_topics_];
        keyword_ptr = keyword_ptr_;
        num_keyword = num_keyword_;
        log_topic = new float[num_topics_];
        senlik = 100;
        log_neighbortopics = new float[2 * neighbor_ * num_topics_];
        init();
    }
    void init();
    ~Sentence(){
        if(xi) delete[] xi;
        if(log_gamma) delete[] log_gamma;
        if(words_ptr) delete[] words_ptr;
        if(words_cnt_ptr) delete[] words_cnt_ptr;
        if(log_topic) delete[] log_topic;
        if(log_neighbortopics) delete[] log_neighbortopics;
        if(keyword_ptr) delete[] keyword_ptr;
    }
};

struct Document{
    float * log_doctopic; //?
    float * log_docTopicMatrix; //?
    int num_total_words; //?
    int num_topics; //?
    float doclik; //?
    int num_sentences;
    struct Sentence ** sentences;
    int neighbor;
    Document(int num_total_words_, int num_topics_, int neighbor_, int num_sentences_){
        num_topics = num_topics_;
        num_total_words = num_total_words_;
        doclik = 100;
        neighbor = neighbor_;
        num_sentences = num_sentences_;
        log_doctopic = new float[num_topics_];
        log_docTopicMatrix = new float[((neighbor_*2)+num_sentences_)*num_topics_];
        sentences = new Sentence* [num_sentences_];
        init();
    }
    Document(int num_total_words_, int num_topics_, int neighbor_, int num_sentences_, struct Sentence** sentence_){
        num_topics = num_topics_;
        num_total_words = num_total_words_;
        doclik = 0.0;
        neighbor = neighbor_;
        num_sentences = num_sentences_;
        log_doctopic = new float[num_topics_];
        log_docTopicMatrix = new float[((neighbor_*2)+num_sentences_)*num_topics_];
        sentences = sentence_;
        init();
    }
    ~Document(){
        if(log_doctopic) delete [] log_doctopic;
        if(sentences) delete [] sentences;
        if(log_docTopicMatrix) delete [] log_docTopicMatrix;
    }
    void init();
};

struct Model {
    int num_words;
    int num_topics;
    int neighbor;
    int num_all_keywords;
    int num_docs;

    float* pi; //the hyper parameter of attention
    float* log_beta; //topic distribution over the words
    float* log_phi; //topic distribution of the keywords

    Model(int num_docs_, int num_topics_, int num_words_, int num_all_keywords_,
          int neighbor_, Model* init_model = NULL){

        num_docs = num_docs_;
        num_words = num_words_;
        neighbor = neighbor_;
        num_topics = num_topics_;
        num_all_keywords = num_all_keywords_;
        pi = new float[num_all_keywords_+2*neighbor_];

        log_beta = new float[(long long)num_docs_ * num_words_];
        log_phi = new float[(long long)num_all_keywords_ * num_docs_];

        //printf("Now initialize the model...");
        init(init_model);
    }
    void init(Model* init_model=NULL);

    Model(char* pifile, char* phifile, char* betafile, int num_topics_, int neighbor_, int num_words_, int num_all_keywords_, int num_docs_){

        neighbor = neighbor_;
        num_topics = num_topics_;
        num_words = num_words_;
        num_all_keywords = num_all_keywords_;
        num_docs = num_docs_;

        pi = load_mat(pifile, num_all_keywords+2*neighbor, 1);
        log_beta = load_mat(betafile, num_topics, num_words);
        log_phi = load_mat(phifile, num_all_keywords, num_topics);

    }
    Model(char* model_root, char* prefix) {
        read_model_info(model_root);
        char pi_file[1000];
        sprintf(pi_file, "%s/%s.pi", model_root, prefix);
        char beta_file[1000];
        sprintf(beta_file,"%s/%s.log_beta",model_root,prefix);
        char phi_file[1000];
        sprintf(phi_file,"%s/%s.log_phi",model_root,prefix);

        pi = load_mat(pi_file, num_all_keywords+2*neighbor, 1);
        log_beta = load_mat(beta_file, num_topics, num_words);
        log_phi = load_mat(phi_file, num_all_keywords, num_topics);
    }
    ~Model() {
        if(pi) delete[] pi;
        if(log_phi) delete[] log_phi;
        if(log_phi) delete[] log_beta;
    }
    void read_model_info(char* model_root);
    float* load_mat(char* filename,int row,int col);
};

struct Configuration {
    float pi_learn_rate;
    int max_pi_iter;
    float pi_min_eps;

    float xi_learn_rate;
    int max_xi_iter;
    float xi_min_eps;

    int max_em_iter;
    int num_threads;

    int sen_max_var_iter;
    int doc_max_var_iter;

    float sen_var_converence;
    float doc_var_converence;
    float em_converence;

    int num_topics;
    int neighbor;
    int num_words;
    int num_all_keywords;


    Configuration(char* settingfile) {
        pi_learn_rate = 0.1;
        max_pi_iter = 100;
        pi_min_eps = 1e-5;

        max_xi_iter = 100;
        xi_learn_rate = 1;
        xi_min_eps = 1e-5;

        max_em_iter = 30;
        num_threads = 1;

        sen_var_converence = 1e-5;
        sen_max_var_iter = 30;

        doc_var_converence = 1e-5;
        doc_max_var_iter = 10;

        em_converence = 1e-4;

        num_topics = 10;
        neighbor = 1;

        num_words = 0;
        num_all_keywords = 0;

        if(settingfile) read_settingfile(settingfile);
    }
    void read_settingfile(char* settingfile);
};


Document ** readData(char* filename, int num_topics,int& num_words, int& num_docs,
                     int& num_all_words, int& num_keywords, int& neighbor);

void printParameters(Document** corpus, int num_round, char* model_root, Model* model);
void readinitbeta(Model* model, char* beta_file);
void readinitphi(Model* model, char* phi_file);
void printtestresults(Document** corpus, int num_round, char* model_root, Model* model);


#endif //ESE_DATA_H
