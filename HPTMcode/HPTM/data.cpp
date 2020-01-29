//
// Created by shuangyinli on 2019-04-14.
//

#include "data.h"
#include "util.h"

Document ** readData(char* filename, int num_topics,int& num_words, int& num_docs,
                     int& num_all_words, int& num_keywords, int& neighbor){
    num_words = 0; //keep the total number words in dictionary
    num_docs = 0; //keep the total number documents in the corpus
    num_all_words = 0; // keep the total number words
    num_keywords = 0; // keep the total number of keywords.
    FILE* fp = fopen(filename,"r"); //calcaulte the file line num

    ifstream infile(filename);
    string buffer;

    while(!infile.eof()){
        getline(infile,buffer);
        num_docs++;
    }
    num_docs--;
    //printf("num_docs: %d \n ",num_docs);

    char str[10];
    Document ** corpus = new Document * [num_docs];

    num_docs = 0;
    int doc_num_sentences;

    while(fscanf(fp,"%d",&doc_num_sentences) != EOF) {
        int doc_num_all_words = 1;

        Sentence ** sentences = new Sentence * [doc_num_sentences];
        for(int i = 0; i < doc_num_sentences; i++){
            fscanf(fp,"%s",str); //read #
            int keyword_num_inSen;
            fscanf(fp, "%d", &keyword_num_inSen);
            int* keyword_ptr = new int[keyword_num_inSen];
            for(int kw=0; kw< keyword_num_inSen; kw++){
                fscanf(fp, "%d", &keyword_ptr[kw]);
                num_keywords = num_keywords > keyword_ptr[kw] ? num_keywords:keyword_ptr[kw];
            }

            int word_num_inSen;

            fscanf(fp, "%s", str); //read @
            fscanf(fp, "%d", &word_num_inSen);

            int* words_ptr = new int[word_num_inSen];
            int* words_cnt_ptr = new int [word_num_inSen];

            for(int w=0; w<word_num_inSen; w++){
                fscanf(fp,"%d:%d", &words_ptr[w],&words_cnt_ptr[w]);
                num_words = num_words < words_ptr[w]?words_ptr[w]:num_words;
                doc_num_all_words += words_cnt_ptr[w];

            }
            sentences[i] = new Sentence(words_ptr, words_cnt_ptr, keyword_ptr, word_num_inSen, keyword_num_inSen, num_topics, neighbor);
        }
        corpus[num_docs++]  = new Document(doc_num_all_words, num_topics, neighbor, doc_num_sentences, sentences);

        if (num_docs % 1000 == 0){
            printf("Read %d docs ... \n",num_docs);
        }

        num_all_words +=doc_num_all_words;
    }
    fclose(fp);
    //the keywords and words lists are numbered from 0 !!!
    num_keywords ++;
    num_words ++;
    //printf("num_docs: %d \nnum_words: %d\nnum_keywords: %d\n",num_docs,num_keywords, num_keywords);
    return corpus;
}

void readinitbeta(Model* model, char* beta_file){
    FILE* fp_beta = fopen(beta_file, "r");
    float * log_beta_ = model->log_beta;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    for (int i = 0; i < num_topics; i++) {
        for (int j = 0; j < num_words; j++) {
            fscanf(fp_beta, "%f", &log_beta_[i * num_words + j]);
        }
    }
    fclose(fp_beta);
}

void readinitphi(Model* model, char* phi_file){
    int num_topics = model->num_topics;
    FILE* fp_phi = fopen(phi_file, "r");
    float * log_phi_ = model->log_phi;
    int num_all_keywords = model->num_all_keywords;
    for(int i = 0; i < num_all_keywords; i++){
        for(int j = 0; j < num_topics; j++){
            fscanf(fp_phi, "%f", &log_phi_[i * num_topics + j]);
        }
    }
    fclose(fp_phi);
}


void Configuration::read_settingfile(char* settingfile){
    FILE* fp = fopen(settingfile,"r");
    char key[100];
    //char test_action[100];
    while (fscanf(fp,"%s",key)!=EOF){
        if (strcmp(key,"phi_learn_rate")==0) {
            fscanf(fp,"%f",&pi_learn_rate);
            continue;
        }
        if (strcmp(key,"max_phi_iter") == 0) {
            fscanf(fp,"%d",&max_pi_iter);
            continue;
        }
        if (strcmp(key,"phi_min_eps") == 0) {
            fscanf(fp,"%f",&pi_min_eps);
            continue;
        }
        if (strcmp(key,"xi_learn_rate") == 0) {
            fscanf(fp,"%f",&xi_learn_rate);
            continue;
        }
        if (strcmp(key,"max_xi_iter") == 0) {
            fscanf(fp,"%d",&max_xi_iter);
            continue;
        }
        if (strcmp(key,"xi_min_eps") == 0) {
            fscanf(fp,"%f",&xi_min_eps);
            continue;
        }
        if (strcmp(key,"max_em_iter") == 0) {
            fscanf(fp,"%d",&max_em_iter);
            continue;
        }
        if (strcmp(key,"num_threads") == 0) {
            fscanf(fp, "%d", &num_threads);
            continue;
        }
        if (strcmp(key, "sen_var_converence") == 0) {
            fscanf(fp, "%f", &sen_var_converence);
            continue;
        }
        if (strcmp(key, "sen_max_var_iter") == 0) {
            fscanf(fp, "%d", &sen_max_var_iter);
            continue;
        }
        if (strcmp(key, "doc_var_converence") == 0) {
            fscanf(fp, "%f", &doc_var_converence);
            continue;
        }
        if (strcmp(key, "doc_max_var_iter") == 0) {
            fscanf(fp, "%d", &doc_max_var_iter);
            continue;
        }
        if (strcmp(key, "em_converence") == 0) {
            fscanf(fp, "%f", &em_converence);
            continue;
        }
        if (strcmp(key, "num_topics") == 0) {
            fscanf(fp, "%d", &num_topics);
            continue;
        }
        if (strcmp(key, "neighbor") == 0) {
            fscanf(fp, "%d", &neighbor);
            continue;
        }
        if (strcmp(key, "num_all_keywords") == 0) {
            fscanf(fp, "%d", &num_all_keywords);
            continue;
        }
        if (strcmp(key, "num_words") == 0) {
            fscanf(fp, "%d", &num_words);
            continue;
        }

    }
}
void Model::init(Model* init_model) {
    if (init_model) {
        for (int k = 0; k < num_topics; k++) {
            for (int i = 0; i < num_words; i++) log_beta[k*num_words + i] = init_model->log_beta[k*num_words + i];
        }
        for(int i=0; i< num_all_keywords; i++){
            pi[i] = init_model->pi[i];
            for (int k = 0; k < num_topics; k++) {
                log_phi[i*num_topics + k] = init_model->log_phi[i*num_topics + k];
            }
        }

        for(int i=0; i< 2*neighbor; i++){
            pi[i+num_all_keywords] = init_model->pi[i+num_all_keywords];
        }

        return;
    }
    for (int i = 0; i < num_all_keywords; i++) {
        pi[i] = util::random() * 0.5 + 1;
        double temp = 0;
        for (int k = 0; k < num_topics; k++) {
            double v = util::random();
            temp += v;
            log_phi[i * num_topics + k] = v;
        }
        for (int k = 0; k < num_topics; k++) {
            log_phi[i * num_topics + k] = log(log_phi[i * num_topics + k] / temp);
        }
    }

    for(int i=0; i< 2*neighbor; i++){
        pi[i+num_all_keywords] = util::random() * 0.5 + 1;
    }

    for (int k = 0; k < num_topics; k++) {
        for (int i = 0; i < num_words; i++){
            log_beta[k*num_words + i] = log(1.0 / num_words);
        }
    }

}

void Document::init(){
    float total = 0;
    for (int i = 0; i < num_topics; i++) {
        float temrandom = rand()/(RAND_MAX+1.0);
        log_doctopic[i] = temrandom;
        total += temrandom;
    }
    for (int i = 0; i < num_topics; i++) {
        log_doctopic[i] = log(log_doctopic[i]/total);
    }

    for(int i=0; i<(2*neighbor)+num_sentences; i++){
        total = 0;
        for(int j=0; j<num_topics; j++){
            float temrandom = rand()/(RAND_MAX+1.0);
            log_docTopicMatrix[i*num_topics + j] = temrandom;
            total += temrandom;
        }
        for(int j=0; j<num_topics; j++){
            log_docTopicMatrix[i*num_topics + j] = log(log_docTopicMatrix[i*num_topics + j]/ total);
        }
    }
}

void Sentence::init() {

    for (int i = 0; i < 2 * neighbor + num_keyword; i++) {
        xi[i] = util::random();
    }

    float total = 0;

    for (int i = 0; i < num_words; i++) {
        total = 0;
        for (int k = 0; k < num_topics; k++){
            float temrandom = rand()/(RAND_MAX+1.0);
            log_gamma[i * num_topics + k] = temrandom;
            total += temrandom;
        }
        for (int k = 0; k < num_topics; k++){
            log_gamma[i * num_topics + k] = log(log_gamma[i * num_topics + k] / total);
        }
    }

    total = 0;
    for (int i = 0; i < num_topics; i++) {
        float temrandom = rand()/(RAND_MAX+1.0);
        log_topic[i] = temrandom;
        total += temrandom;
    }
    for (int i = 0; i < num_topics; i++) {
        log_topic[i] = log(log_topic[i]/ total);
    }

    for (int i = 0; i < 2 * neighbor; i++) {
        total = 0;
        for (int j = 0; j < num_topics; j++) {
            float temrandom = rand()/(RAND_MAX+1.0);
            log_neighbortopics[i * num_topics + j] = temrandom;
            total += temrandom;
        }
        for (int j = 0; j < num_topics; j++) {
            log_neighbortopics[i * num_topics + j] = log(log_neighbortopics[i * num_topics + j] / total);
        }
    }
}

void Model::read_model_info(char* model_root) {
    char filename[1000];
    sprintf(filename, "%s/model.info",model_root);
    printf("%s\n",filename);
    FILE* fp = fopen(filename,"r");
    char str[100];
    int value;
    while (fscanf(fp,"%s%d",str,&value)!=EOF) {
        if (strcmp(str, "num_words:") == 0)num_words = value;
        if (strcmp(str, "num_topics:") == 0)num_topics = value;
        if (strcmp(str, "num_docs:") == 0) num_docs = value;
    }
    printf("num_words: %d\nnum_topics: %d\n",num_words, num_topics);
    fclose(fp);
}

float* Model::load_mat(char* filename, int row, int col) {
    FILE* fp = fopen(filename,"r");
    float* mat = new float[row * col];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fscanf(fp, "%f", &mat[i*col+j]);
        }
    }
    fclose(fp);
    return mat;
}


void print_mat(float* mat, int row, int col, char* filename) {
    FILE* fp = fopen(filename,"w");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fprintf(fp,"%f ",mat[i*col + j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

void printParameters(Document** corpus, int num_round, char* model_root, Model* model) {
    char pi_file[1000];
    char phi_file[1000];
    char beta_file[1000];
    char topic_dis_file[1000];
    char sentence_topic_dis_file[1000];
    char liks_file[1000];
    if (num_round != -1) {
        sprintf(pi_file, "%s/%03d.pi", model_root, num_round);
        sprintf(phi_file, "%s/%03d.phi", model_root, num_round);
        sprintf(beta_file, "%s/%03d.beta", model_root, num_round);
        sprintf(topic_dis_file, "%s/%03d.doc_distribution", model_root, num_round);
        sprintf(sentence_topic_dis_file, "%s/%03d.sentence_distribution", model_root, num_round);
        sprintf(liks_file, "%s/%03d.likehoods", model_root, num_round);
    }
    else {
        sprintf(pi_file, "%s/final.pi", model_root);
        sprintf(phi_file, "%s/final.phi", model_root);
        sprintf(beta_file, "%s/final.beta", model_root);
        sprintf(topic_dis_file,"%s/final.doc_distribution", model_root);
        sprintf(sentence_topic_dis_file, "%s/final.sentence_distribution", model_root);
        sprintf(liks_file, "%s/final.likehoods", model_root);
    }
    print_mat(model->log_beta, model->num_topics, model->num_words, beta_file);
    print_mat(model->log_phi, model->num_all_keywords, model->num_topics, phi_file);


    FILE* pi_fp = fopen(pi_file,"w");

    for(int p=0; p < model->num_all_keywords+2*model->neighbor; p++){
        fprintf(pi_fp, "%lf\n", model->pi[p]);
    }

    fclose(pi_fp);

    //save the topic distribution of all the sentences.
    int num_docs = model->num_docs;
    FILE* topic_dis_fp = fopen(topic_dis_file,"w");
    FILE* topic_dis_fp_sentences = fopen(sentence_topic_dis_file,"w");
    FILE* liks_fp = fopen(liks_file, "w");
    for (int d = 0; d < num_docs; d++) {
        fprintf(liks_fp, "%lf\n", corpus[d]->doclik);
        Document* doc = corpus[d];
        fprintf(topic_dis_fp, "%lf", doc->log_doctopic[0]);
        for (int k = 1; k < doc->num_topics; k++)fprintf(topic_dis_fp, " %lf", doc->log_doctopic[k]);
        fprintf(topic_dis_fp, "\n");

        int num_sentences = doc->num_sentences;
        for(int s=0; s<num_sentences; s++){
            Sentence * sentence = doc->sentences[s];
            fprintf(topic_dis_fp_sentences, "%lf", sentence->log_topic[0]);
            for(int k = 1; k < sentence->num_topics; k++){
                fprintf(topic_dis_fp_sentences, " %lf", sentence->log_topic[k]);
            }
            fprintf(topic_dis_fp_sentences, " # ");
        }
        fprintf(topic_dis_fp_sentences, "\n");
    }
    fclose(topic_dis_fp);
    fclose(liks_fp);
    fclose(topic_dis_fp_sentences);
}

void printtestresults(Document** corpus, int num_round, char* model_root, Model* model) {

    char topic_dis_file[1000];
    char sentence_topic_dis_file[1000];
    char attentionfile[1000];

    if (num_round != -1) {
        sprintf(topic_dis_file, "%s/%03d.doc_distribution", model_root, num_round);
        sprintf(sentence_topic_dis_file, "%s/%03d.sentence_distribution", model_root, num_round);
        sprintf(attentionfile, "%s/%03d.attentions", model_root, num_round);
    }
    else {
        sprintf(topic_dis_file,"%s/final.doc_distribution", model_root);
        sprintf(sentence_topic_dis_file, "%s/final.sentence_distribution", model_root);
        sprintf(attentionfile, "%s/final.attentions", model_root);
    }

    int num_docs = model->num_docs;
    FILE* topic_dis_fp = fopen(topic_dis_file,"w");
    FILE* topic_dis_fp_sentences = fopen(sentence_topic_dis_file,"w");
    FILE* attentions_fp_sentences = fopen(attentionfile,"w");


    for (int d = 0; d < num_docs; d++) {

        Document* doc = corpus[d];
        fprintf(topic_dis_fp, "%f", doc->log_doctopic[0]);
        for (int k = 1; k < doc->num_topics; k++)fprintf(topic_dis_fp, " %f", doc->log_doctopic[k]);
        fprintf(topic_dis_fp, "\n");

        int num_sentences = doc->num_sentences;
        for(int s=0; s<num_sentences; s++){
            Sentence * sentence = doc->sentences[s];
            fprintf(topic_dis_fp_sentences, "%f", sentence->log_topic[0]);
            for(int k = 1; k < sentence->num_topics; k++){
                fprintf(topic_dis_fp_sentences, " %f", sentence->log_topic[k]);
            }
            fprintf(topic_dis_fp_sentences, " #");


            fprintf(attentions_fp_sentences, "%d:%f ", sentence->keyword_ptr[0],sentence->xi[0]);

            for(int a =1; a < sentence->num_keyword; a++){
                fprintf(attentions_fp_sentences, "%d:%f ", sentence->keyword_ptr[a],sentence->xi[a]);
            }

            fprintf(attentions_fp_sentences, "n%d:%f ", 0,sentence->xi[0+sentence->num_keyword]);


            for(int a =1; a < 2*doc->neighbor; a++){
                fprintf(attentions_fp_sentences, "n%d:%f ", a,sentence->xi[a+sentence->num_keyword]);
            }
            fprintf(attentions_fp_sentences, "# ");
        }

        fprintf(topic_dis_fp_sentences, "\n");
        fprintf(attentions_fp_sentences, "\n");

    }


    fclose(topic_dis_fp);
    fclose(topic_dis_fp_sentences);
    fclose(attentions_fp_sentences);

}
