//
// Created by shuangyinli on 2019-04-14.
//

#include "estimate.h"


float corpuslikelihood(Document** corpus, Model* model) {
    int num_docs = model->num_docs;
    float lik = 0.0;
    for (int d = 0; d < num_docs; d++){
        float temp_lik = compute_document_likelihood(corpus[d],model);
        lik += temp_lik;
        corpus[d]->doclik = temp_lik;
    }
    return lik;
}

float compute_document_likelihood(Document* doc, Model* model) {
    int num_sentence = doc->num_sentences;
    float lik = 0.0;
    for (int s = 0; s < num_sentence; s++) {
        float temp_lik = compute_sentence_likehood(doc->sentences[s],model);
        lik += temp_lik;
        doc->sentences[s]->senlik = temp_lik;
    }
    return lik;
}

float compute_sentence_likehood(Sentence* sentence, Model* model) {
    float* log_phi = model->log_phi;
    float* log_beta = model->log_beta;
    int num_topics = model->num_topics;
    int num_all_words = model->num_words;
    memset(sentence->log_topic, 0, sizeof(float) * num_topics);
    bool* reset_log_semantics = new bool[num_topics];
    memset(reset_log_semantics, false, sizeof(bool) * num_topics);
    float sigma_xi = 0;
    float* xi = sentence->xi;
    int sen_num_keyswords = sentence->num_keyword;
    int lenth_xi = sen_num_keyswords + 2 * sentence->neighbor;
    float lik = 0.0;

    //update the log topic of the current sentence
    for (int i = 0; i < lenth_xi; i++) {
        sigma_xi += xi[i];
    }

    for (int i = 0; i < sen_num_keyswords; i++) {
        int kid = sentence->keyword_ptr[i];
        for (int k = 0; k < num_topics; k++) {
            if (!reset_log_semantics[k]) {
                sentence->log_topic[k] = log_phi[kid * num_topics + k] + log(xi[i]) - log(sigma_xi);
                reset_log_semantics[k] = true;
            }
            else sentence->log_topic[k] = util::log_sum(sentence->log_topic[k], log_phi[kid * num_topics + k] + log(xi[i]) - log(sigma_xi));
        }
    }

    //TODo error, should add the weights of the neighbors. Done
    for(int i = 0; i < 2 * sentence->neighbor; i++){
        for (int k = 0; k < num_topics; k++) {
            sentence->log_topic[k] = util::log_sum(sentence->log_topic[k],  sentence->log_neighbortopics[i*num_topics +k] + log(xi[i+sen_num_keyswords]) - log(sigma_xi));
        }
    }

    //compute the log likelihood
    int sen_num_words = sentence->num_words;
    for (int i = 0; i < sen_num_words; i++) {
        float temp = 0;
        int wordid = sentence->words_ptr[i];
        temp = sentence->log_topic[0] + log_beta[wordid];
        for (int k = 1; k < num_topics; k++) temp = util::log_sum(temp, sentence->log_topic[k] + log_beta[k * num_all_words + wordid]);
        lik += temp * sentence->words_cnt_ptr[i];
    }
    delete[] reset_log_semantics;
    sentence->senlik = lik;
    return lik;
}


void initPi(float* pi, int num) {
    for (int i = 0; i < num; i++) {
        pi[i] = util::random() * 2;
    }
}

float getPiFunction(Document** corpus, Model* model, int num_docs) {
    float pi_function_value = 0.0;
    //int len_pi = model->num_all_keywords+2*model->neighbor;
    //int num_docs = model->num_docs;
    float* pi = model->pi;
    for (int d = 0; d < num_docs; d++) {
        Document* doc = corpus[d];
        for(int s=0; s< doc->num_sentences; s++){

            Sentence* sentence = doc->sentences[s];
            float sigma_pi = 0.0;
            float sigma_xi = 0.0;


            for (int i = 0; i < 2*model->neighbor +sentence->num_keyword; i++) {
                sigma_xi += sentence->xi[i];
            }
            for (int i = 0; i < sentence->num_keyword; i++) {
                sigma_pi += pi[sentence->keyword_ptr[i]];
            }
            for (int i = 0; i < 2*model->neighbor; i++) {
                sigma_pi += pi[i+model->num_all_keywords];
            }

            pi_function_value += util::log_gamma(sigma_pi);

            for (int i = 0; i < sentence->num_keyword; i++) {
                int keyword_id = sentence->keyword_ptr[i];
                pi_function_value -= util::log_gamma(pi[keyword_id]);
                pi_function_value += (pi[keyword_id] - 1) * (util::digamma(sentence->xi[i]) - util::digamma(sigma_xi));
            }

            for (int i = 0; i < 2*model->neighbor; i++) {
                pi_function_value -= util::log_gamma(pi[i+model->num_all_keywords]);
                pi_function_value += (pi[i+model->num_all_keywords] - 1) * (util::digamma(sentence->xi[i+sentence->num_keyword]) - util::digamma(sigma_xi));
            }
        }
    }
    return pi_function_value;
}

void getDescentPi(Document** corpus, Model* model, float* descent_pi, int num_docs) {

    int num_all_keywords = model->num_all_keywords;
    //int num_docs = model->num_docs;

    memset(descent_pi,0,sizeof(float)* (num_all_keywords + 2*model->neighbor));
    float* pi = model->pi;
    for (int d = 0; d < num_docs; d++) {
        Document* doc = corpus[d];
        for(int s = 0; s<doc->num_sentences; s++){
            Sentence* sentence = doc->sentences[s];
            float sigma_pi = 0.0;
            float sigma_xi = 0.0;
            int sen_num_keyword = sentence->num_keyword;

            for (int i = 0; i < 2*model->neighbor +sentence->num_keyword; i++) {
                sigma_xi += sentence->xi[i];
            }
            for (int i = 0; i < sentence->num_keyword; i++) {
                sigma_pi += pi[sentence->keyword_ptr[i]];
            }
            for (int i = 0; i < 2*model->neighbor; i++) {
                sigma_pi += pi[i+model->num_all_keywords];
            }

            for (int i = 0; i < sen_num_keyword; i++) {
                int keyword_id = sentence->keyword_ptr[i];
                float pis = pi[keyword_id];
                descent_pi[keyword_id] += util::digamma(sigma_pi) - util::digamma(pis) + util::digamma(sentence->xi[i]) - util::digamma(sigma_xi);
            }

            for (int i = 0; i < 2*model->neighbor; i++) {

                descent_pi[i+model->num_all_keywords] += util::digamma(sigma_pi) - util::digamma(pi[i+model->num_all_keywords]) + util::digamma(sentence->xi[i+sentence->num_keyword]) - util::digamma(sigma_xi);
            }

        }
    }
}

void learnPi(Document** corpus, Model* model, Configuration* configuration) {
    int num_round = 0;
    int neighbor = model->neighbor;
    int num_all_keywords = model->num_all_keywords;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    vector<int> poker_index;
    for (int num = 0; num < model->num_docs; num++)
        poker_index.push_back(num);

    shuffle(poker_index.begin(), poker_index.end(),std::default_random_engine(seed));

    set<int> sampledata;
    int num_sampledata = model->num_docs / 10;
    for (int i = 0; i < num_sampledata; i++) {
        int m = poker_index[i];
        sampledata.insert(m);
    }

    Document** samplecorpus = new Document *[num_sampledata];

    set<int>::iterator it;
    int num_samples = 0;
    for(it = sampledata.begin(); it != sampledata.end(); it ++){
        int id = *it;
        samplecorpus[num_samples] = corpus[id];
        num_samples++;
    }

    int len_pi = num_all_keywords+2*neighbor;

    float* last_pi = new float [len_pi];
    float* descent_pi = new float[len_pi];
    float z;
    int num_wait_for_z = 0;
    do {
        initPi(model->pi,len_pi);
        z = getPiFunction(samplecorpus,model,num_sampledata);
        num_wait_for_z ++;
    }
    while ( z < 0 && num_wait_for_z <= 20);
    float last_z;
    float learn_rate = configuration->pi_learn_rate;
    float eps = 1000;
    int max_pi_iter = configuration->max_pi_iter;
    float pi_min_eps = configuration->pi_min_eps;
    bool has_neg_value_flag = false;
    do {
        last_z = z;
        memcpy(last_pi,model->pi,sizeof(float) * len_pi);
        getDescentPi(samplecorpus,model,descent_pi,num_sampledata);
        for (int i = 0; !has_neg_value_flag && i < len_pi; i++) {
            model->pi[i] += learn_rate * descent_pi[i];
            if (model->pi[i] < 0) has_neg_value_flag = true;
        }
        if (has_neg_value_flag || last_z > (z=getPiFunction(samplecorpus,model,num_sampledata))) {
            learn_rate *= 0.5;
            z = last_z;
            //for ( int i = 0; i < num_labels; i++) pi[i] = last_pi[i];
            memcpy(model->pi,last_pi,sizeof(float) * len_pi);
            eps = 1000.0;
        }
        else eps = util::norm2(last_pi, model->pi, len_pi);
        num_round += 1;
    }
    while (num_round < max_pi_iter && eps > pi_min_eps);
    delete[] last_pi;
    delete[] descent_pi;
}

void* thread_learn_phi_beta(void* thread_data_theta_beta) {
    Thread_Data_phi_beta* thread_data_ptr = (Thread_Data_phi_beta*) thread_data_theta_beta;
    Document ** corpus = thread_data_ptr->corpus;
    int start = thread_data_ptr->start;
    int end = thread_data_ptr->end;
    Model* thread_model = thread_data_ptr->thread_model;

    int num_topics = thread_model->num_topics;
    int num_all_words = thread_model->num_words;

    bool* reset_theta_flag = new bool[thread_model->num_all_keywords * num_topics];
    memset(reset_theta_flag, 0, sizeof(bool) * thread_model->num_all_keywords * num_topics);
    bool* reset_beta_flag = new bool[num_topics * thread_model->num_words];
    memset(reset_beta_flag, 0, sizeof(bool) * num_topics * thread_model->num_words);

    for (int i = start; i < end; i++) {
        Document* doc = corpus[i];
        for(int s = 0; s<doc->num_sentences; s++){
            Sentence * sentence = doc->sentences[s];
            int num_keyword = sentence->num_keyword;
            int num_words = sentence->num_words;
            float sigma_xi = 0;
            for (int i = 0; i < num_keyword; i++) sigma_xi += sentence->xi[i];
            for (int i = 0; i < num_keyword; i++) {
                int keyword_id = sentence->keyword_ptr[i];
                for (int k = 0; k < num_topics; k++) {
                    for (int j = 0; j < num_words; j++) {
                        if (!reset_theta_flag[keyword_id * num_topics + k]) {
                            reset_theta_flag[keyword_id * num_topics + k] = true;
                            thread_model->log_phi[keyword_id * num_topics + k] = log(sentence->words_cnt_ptr[j]) + sentence->log_gamma[j * num_topics + k] + log(sentence->xi[i]) - log(sigma_xi);
                        }else {
                            thread_model->log_phi[keyword_id * num_topics + k] = util::log_sum(thread_model->log_phi[keyword_id * num_topics + k], log(sentence->words_cnt_ptr[j]) +sentence->log_gamma[j * num_topics + k] + log(sentence->xi[i]) - log(sigma_xi));
                        }
                    }

                }
            }
            for (int k = 0; k < num_topics; k++) {
                for (int i = 0; i < num_words; i++) {
                    int wordid = sentence->words_ptr[i];
                    if (!reset_beta_flag[k * num_all_words + wordid]) {
                        reset_beta_flag[k * num_all_words + wordid] = true;
                        thread_model->log_beta[k * num_all_words + wordid] = log(sentence->words_cnt_ptr[i]) + sentence->log_gamma[i*num_topics + k];
                    }else {
                        thread_model->log_beta[k * num_all_words + wordid] = util::log_sum(thread_model->log_beta[k * num_all_words + wordid], sentence->log_gamma[i*num_topics + k] + log(sentence->words_cnt_ptr[i]));
                    }
                }
            }

        }

    }

    delete[] reset_theta_flag;
    delete[] reset_beta_flag;
    return NULL;
}

void run_thread_learn_phi_beta(Document** corpus, Model* model, Configuration* configuration) {
    int num_threads = configuration->num_threads;
    pthread_t* pthread_ts = new pthread_t[num_threads];
    int num_docs = model->num_docs;
    int num_topics = model->num_topics;
    int neighbor = model->neighbor;
    int num_per_threads = num_docs/num_threads;
    int i;
    int j;
    int k;

    //
    Thread_Data_phi_beta** thread_datas_phi_beta = new Thread_Data_phi_beta* [num_threads];
    for (i = 0; i < num_threads - 1; i++) {
        Model * thread_model = new Model(num_docs,num_topics, model->num_words,model->num_all_keywords,neighbor);
        thread_datas_phi_beta[i] = new Thread_Data_phi_beta(corpus, i * num_per_threads, (i+1)*num_per_threads, thread_model);
        pthread_create(&pthread_ts[i], NULL, thread_learn_phi_beta, (void*) thread_datas_phi_beta[i]);
    }

    Model * thread_model = new Model(num_docs,num_topics, model->num_words,model->num_all_keywords, neighbor);
    thread_datas_phi_beta[i] = new Thread_Data_phi_beta(corpus, i * num_per_threads, num_docs, thread_model);
    pthread_create(&pthread_ts[i], NULL, thread_learn_phi_beta, (void*) thread_datas_phi_beta[i]);
    for (i = 0; i < num_threads; i++) pthread_join(pthread_ts[i],NULL);

    // add all the thread_models

    int num_all_words = model->num_words;
    int num_all_keywords = model->num_all_keywords;
    bool* reset_theta_flag = new bool[thread_model->num_all_keywords * num_topics];
    memset(reset_theta_flag, 0, sizeof(bool) * thread_model->num_all_keywords * num_topics);
    bool* reset_beta_flag = new bool[num_topics * thread_model->num_words];
    memset(reset_beta_flag, 0, sizeof(bool) * num_topics * thread_model->num_words);

    for (i = 0; i < num_threads; i++){
        Model * thread_model = thread_datas_phi_beta[i]->thread_model;
        //phi
        for (j = 0; j <num_all_keywords; j++){
            for(k = 0; k< num_topics; k++){
                if (!reset_theta_flag[j * num_topics + k]) {
                    reset_theta_flag[j * num_topics + k] = true;
                    model->log_phi[j*num_topics + k] = thread_model->log_phi[j*num_topics + k];
                }else{
                    model->log_phi[j*num_topics + k] = util::log_sum(model->log_phi[j*num_topics + k], thread_model->log_phi[j*num_topics + k]);
                }
            }
        }
        //beta
        for (int k = 0; k < num_topics; k++) {
            for (int w = 0; w < num_all_words; w++) {
                if (!reset_beta_flag[k * num_all_words + w]) {
                    reset_beta_flag[k * num_all_words + w] = true;
                    model->log_beta[k * num_all_words + w] = thread_model->log_beta[k * num_all_words + w];
                }
                else {
                    model->log_beta[k * num_all_words + w] = util::log_sum(model->log_beta[k * num_all_words + w], thread_model->log_beta[k * num_all_words + w]);
                }

            }
        }

    }

    //normalize_log_matrix_rows
    normalize_log_matrix_rows(model->log_phi, model->num_all_keywords, num_topics);
    normalize_log_matrix_rows(model->log_beta, num_topics, model->num_words);

    for (i = 0; i < num_threads; i++){
        delete thread_datas_phi_beta[i]->thread_model;
        delete thread_datas_phi_beta[i];
    }
    delete[] thread_datas_phi_beta;
}


void normalize_matrix_rows(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float temp = 0;
        for (int j = 0; j < cols; j++) temp += mat[ i * cols + j];
        for (int j = 0; j < cols; j++) {
            mat[i*cols +j] /= temp;
            if (mat[i*cols + j] == 0)mat[i*cols + j] = 1e-300;
        }
    }
}

void normalize_log_matrix_rows(float* log_mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float temp = log_mat[ i * cols];
        /*if (isnan(temp) || isnan(-temp)) {
            printf("normalize nan\n");
        }*/
        for (int j = 1; j < cols; j++) temp = util::log_sum(temp, log_mat[i * cols + j]);
        /*if (isnan(-temp) || isnan(temp)) {
            printf("normalize nan\n");
        }*/
        for (int j = 0; j < cols; j++) log_mat[i*cols + j] -= temp;
    }
}

void* thread_learn_phi(void* thread_data_theta) {
    Thread_Data_phi* thread_data_ptr = (Thread_Data_phi*) thread_data_theta;
    Document ** corpus = thread_data_ptr->corpus;
    int start = thread_data_ptr->start;
    int end = thread_data_ptr->end;
    Model* thread_model = thread_data_ptr->thread_model;

    int num_topics = thread_model->num_topics;
    int num_all_words = thread_model->num_words;

    bool* reset_theta_flag = new bool[thread_model->num_all_keywords * num_topics];
    memset(reset_theta_flag, 0, sizeof(bool) * thread_model->num_all_keywords * num_topics);

    for (int i = start; i < end; i++) {
        Document* doc = corpus[i];
        for(int s = 0; s<doc->num_sentences; s++){
            Sentence * sentence = doc->sentences[s];
            int num_keyword = sentence->num_keyword;
            int num_words = sentence->num_words;
            float sigma_xi = 0;
            for (int i = 0; i < num_keyword; i++) sigma_xi += sentence->xi[i];
            for (int i = 0; i < num_keyword; i++) {
                int keyword_id = sentence->keyword_ptr[i];
                for (int k = 0; k < num_topics; k++) {
                    for (int j = 0; j < num_words; j++) {
                        if (!reset_theta_flag[keyword_id * num_topics + k]) {
                            reset_theta_flag[keyword_id * num_topics + k] = true;
                            thread_model->log_phi[keyword_id * num_topics + k] = log(sentence->words_cnt_ptr[j]) + sentence->log_gamma[j * num_topics + k] + log(sentence->xi[i]) - log(sigma_xi);
                        }else {
                            thread_model->log_phi[keyword_id * num_topics + k] = util::log_sum(thread_model->log_phi[keyword_id * num_topics + k], log(sentence->words_cnt_ptr[j]) +sentence->log_gamma[j * num_topics + k] + log(sentence->xi[i]) - log(sigma_xi));
                        }
                    }

                }
            }

        }

    }

    delete[] reset_theta_flag;
    return NULL;
}


void run_thread_learn_phi(Document** corpus, Model* model, Configuration* configuration) {
    int num_threads = configuration->num_threads;
    pthread_t* pthread_ts = new pthread_t[num_threads];
    int num_docs = model->num_docs;
    int num_topics = model->num_topics;
    int neighbor = model->neighbor;
    int num_per_threads = num_docs/num_threads;
    int i;
    int j;
    int k;

    //
    Thread_Data_phi** thread_datas_phi = new Thread_Data_phi* [num_threads];
    for (i = 0; i < num_threads - 1; i++) {
        Model * thread_model = new Model(num_docs,num_topics, model->num_words,model->num_all_keywords,neighbor);
        thread_datas_phi[i] = new Thread_Data_phi(corpus, i * num_per_threads, (i+1)*num_per_threads, thread_model);
        pthread_create(&pthread_ts[i], NULL, thread_learn_phi, (void*) thread_datas_phi[i]);
    }

    Model * thread_model = new Model(num_docs,num_topics, model->num_words,model->num_all_keywords, neighbor);
    thread_datas_phi[i] = new Thread_Data_phi(corpus, i * num_per_threads, num_docs, thread_model);
    pthread_create(&pthread_ts[i], NULL, thread_learn_phi, (void*) thread_datas_phi[i]);
    for (i = 0; i < num_threads; i++) pthread_join(pthread_ts[i],NULL);

    // add all the thread_models

    int num_all_words = model->num_words;
    int num_all_keywords = model->num_all_keywords;
    bool* reset_theta_flag = new bool[thread_model->num_all_keywords * num_topics];
    memset(reset_theta_flag, 0, sizeof(bool) * thread_model->num_all_keywords * num_topics);

    for (i = 0; i < num_threads; i++){
        Model * thread_model = thread_datas_phi[i]->thread_model;
        //phi
        for (j = 0; j <num_all_keywords; j++){
            for(k = 0; k< num_topics; k++){
                if (!reset_theta_flag[j * num_topics + k]) {
                    reset_theta_flag[j * num_topics + k] = true;
                    model->log_phi[j*num_topics + k] = thread_model->log_phi[j*num_topics + k];
                }else{
                    model->log_phi[j*num_topics + k] = util::log_sum(model->log_phi[j*num_topics + k], thread_model->log_phi[j*num_topics + k]);
                }
            }
        }

    }

    //normalize_log_matrix_rows
    normalize_log_matrix_rows(model->log_phi, model->num_all_keywords, num_topics);

    for (i = 0; i < num_threads; i++){
        delete thread_datas_phi[i]->thread_model;
        delete thread_datas_phi[i];
    }
    delete[] thread_datas_phi;
}