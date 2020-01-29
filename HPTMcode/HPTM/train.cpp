//
// Created by shuangyinli on 2019-04-14.
//

#include "train.h"

void doInference(Document* doc, Model* model, Configuration* configuration) {
    float doc_lik_old = 0.0;
    float doc_lik = 0.0;
    float doc_var_converence = 1.0;
    int doc_max_var_iter = 0;
    int num_sentence = doc->num_sentences;
    int neighbor = model->neighbor;
    int num_topics = model->num_topics;

    while((doc_var_converence > configuration->doc_var_converence) && (doc_max_var_iter < configuration->doc_max_var_iter)){
        doc_max_var_iter ++;

        for(int s =0; s<num_sentence; s++){
            //for each sentence
            int var_iter = 0;
            float sen_lik_old = 0.0;
            float sen_converged = 1.0;
            float sen_lik=0.0;
            Sentence* current_sentence = doc->sentences[s];

            //init the neighbor topics for each sentence
            for(int i=0; i< neighbor; i++)
                for (int k = 0; k < num_topics; k++) {
                    current_sentence->log_neighbortopics[i * num_topics + k] = doc->log_docTopicMatrix[
                            (s + neighbor - i - 1) * num_topics + k];
                }
            for(int i=0; i< neighbor; i++)
                for (int k = 0; k < num_topics; k++) {
                    current_sentence->log_neighbortopics[(i + neighbor) * num_topics + k] = doc->log_docTopicMatrix[
                            (s + neighbor + i + 1) * num_topics + k];
                }
            float * old_sen_loggamma = new float[current_sentence->num_words * num_topics];
            float * old_sen_log_topics = new float[num_topics];
            float * old_sen_xi = new float[2*neighbor + current_sentence->num_keyword];
            memcpy(old_sen_loggamma,current_sentence->log_gamma,sizeof(float)*current_sentence->num_words*num_topics);
            memcpy(old_sen_log_topics,current_sentence->log_topic, sizeof(float)*num_topics);
            memcpy(old_sen_xi,current_sentence->xi,sizeof(float)*(2*neighbor + current_sentence->num_keyword));

            while ((sen_converged > configuration->sen_var_converence) &&
                   ((var_iter < configuration->sen_max_var_iter || configuration->sen_max_var_iter == -1))) {
                var_iter ++;
                inferenceXi(current_sentence, model, configuration);
                inferenceGamma(current_sentence, model);
                //update the topics of the current sentence in 'inference_sen_likelihood' function
                sen_lik = compute_sentence_likehood(current_sentence,model);
                current_sentence->senlik = sen_lik;
                sen_converged = (sen_lik_old -sen_lik) / sen_lik_old;
                if(sen_converged < 0){
                    memcpy(current_sentence->log_gamma, old_sen_loggamma,sizeof(float)*current_sentence->num_words*num_topics);
                    memcpy(current_sentence->log_topic, old_sen_log_topics,sizeof(float)*num_topics);
                    memcpy(current_sentence->xi, old_sen_xi,sizeof(float)*(2*neighbor + current_sentence->num_keyword));
                    current_sentence->senlik = sen_lik;
                    break;
                }

                memcpy(old_sen_loggamma,current_sentence->log_gamma,sizeof(float)*current_sentence->num_words*num_topics);
                memcpy(old_sen_log_topics,current_sentence->log_topic,sizeof(float)*num_topics);
                memcpy(old_sen_xi,current_sentence->xi,sizeof(float)*(2*neighbor + current_sentence->num_keyword));

                sen_lik_old = sen_lik;
            }

            for(int k=0; k<num_topics; k++){
                //update the log topic distribution of the current sentence
                doc->log_docTopicMatrix[(s+neighbor)*num_topics + k] = current_sentence->log_topic[k];
            }
            delete[] old_sen_loggamma;
            delete[] old_sen_log_topics;
            delete[] old_sen_xi;

            doc_lik += sen_lik;


        }

        doc_var_converence = fabs(doc_lik_old - doc_lik);
        doc_lik_old = doc_lik;
        //update the topic distribution of the document
        //computeDocTopicDistribution(doc, model);
        doc->doclik = doc_lik;

        //update the first and the last blank 'sentences'
        for(int i=0; i< neighbor; i++)
            for (int k = 0; k < num_topics; k++){
                doc->log_docTopicMatrix[i*num_topics + k]=doc->log_docTopicMatrix[neighbor*num_topics + k];
                //doc->log_docTopicMatrix[i*num_topics + k]=doc->log_doctopic[k];
            }

        for(int i=neighbor+doc->num_sentences; i< 2*neighbor+doc->num_sentences; i++){
            for(int k = 0; k < num_topics; k++){
                //doc->log_docTopicMatrix[i*num_topics + k]=doc->log_doctopic[k];
                doc->log_docTopicMatrix[i*num_topics + k]=doc->log_docTopicMatrix[(neighbor + doc->num_sentences -1)*num_topics + k];
            }
        }

        doc_lik = 0.0;

    }

    return;
}

void computeDocTopicDistribution(Document* doc, Model* model){

    int num_sentence = doc->num_sentences;
    int num_topics = model->num_topics;

    for (int s = 0; s < num_sentence; s++) {
        Sentence* sentence = doc->sentences[s];
        int num_all_words =sentence->num_words;
        for(int n = 0; n < num_all_words; n++){

            for(int k=0; k < num_topics; k++){
                doc->log_doctopic[k] += sentence->log_gamma[n*num_topics + k] + log(sentence->words_cnt_ptr[n]);

            }
        }

    }

    float log_sum_topic =0.0;

    for(int k=0; k < num_topics; k++){
        if (k == 0)
            log_sum_topic = doc->log_doctopic[k];
        else
            log_sum_topic =util::log_sum(doc->log_doctopic[k], log_sum_topic);
    }

    for(int k=0; k < num_topics; k++) doc->log_doctopic[k] -= log_sum_topic;

}

void inferenceGamma(Sentence* sentence, Model* model) {
    float* log_phi = model->log_phi;
    float* log_beta = model->log_beta;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    int neighbor = model->neighbor;
    int sen_num_words = sentence->num_words;
    float* log_gamma = sentence->log_gamma;
    float* phi_xi = new float[num_topics];
    float sigma_xi = 0;
    int num_keyword = sentence->num_keyword;

    for (int i = 0; i < 2*neighbor +num_keyword; i++){
        sigma_xi += sentence->xi[i];
    }

    for (int k = 0; k < num_topics; k++){
        float temp = 0;
        for (int i = 0; i < num_keyword; i++) {
            temp += (sentence->xi[i] / sigma_xi) * log_phi[sentence->keyword_ptr[i]*num_topics + k];
        }

        for(int i = 0; i < 2*neighbor; i++){
            temp += (sentence->xi[i+num_keyword] / sigma_xi) * sentence->log_neighbortopics[i*num_topics +k];
        }
        phi_xi[k] = temp;
    }

    // not change
    for (int i = 0; i < sen_num_words; i++) {
        int wordid = sentence->words_ptr[i];
        float sum_log_gamma = 0.0;
        for (int k = 0; k < num_topics; k++) {
            float temp = log_beta[k * num_words + wordid] + phi_xi[k];
            log_gamma[ i * num_topics + k] = temp + log(sentence->words_cnt_ptr[i]);

            if (k == 0) sum_log_gamma = temp;
            else sum_log_gamma = util::log_sum(sum_log_gamma, temp);
        }
        for (int k = 0; k < num_topics; k++)log_gamma[i*num_topics + k] -= sum_log_gamma;
    }

    delete[] phi_xi;
}

void inferenceXi(Sentence* sentence, Model* model,Configuration* configuration) {
    int neighbor = model->neighbor;
    int num_keywords_neighbors = sentence->num_keyword +2*neighbor;
    float* descent_xi = new float[num_keywords_neighbors];
    initXi(sentence->xi,num_keywords_neighbors);
    float z = getXiFunction(sentence,model);
    float learn_rate = configuration->xi_learn_rate;
    float eps = 10000;
    int num_round = 0;
    int max_xi_iter = configuration->max_xi_iter;
    float xi_min_eps = configuration->xi_min_eps;
    float last_z;
    float* last_xi = new float[num_keywords_neighbors];
    float * init_xi = new float[num_keywords_neighbors];
    memcpy(init_xi,sentence->xi,sizeof(float)*num_keywords_neighbors);
    do {
        last_z = z;
        memcpy(last_xi,sentence->xi,sizeof(float)*num_keywords_neighbors);
        getDescentXi(sentence,model,descent_xi);

        bool has_neg_value_flag = false;
        for (int i = 0; !has_neg_value_flag && i < num_keywords_neighbors; i++) {
            sentence->xi[i] += learn_rate * descent_xi[i];
            if (sentence->xi[i] < 0)has_neg_value_flag = true;
            if (isnan(-sentence->xi[i]) || isnan(sentence->xi[i]) || isinf(sentence->xi[i]) || isinf(-sentence->xi[i]) ){

                if (isnan(last_xi[i]) || isnan(last_xi[i])){
                    printf("last xi nan 1 \n");
                    memcpy(sentence->xi,init_xi,sizeof(float)*num_keywords_neighbors);
                } if (isinf(last_xi[i]) || isinf(last_xi[i])){
                    printf("last xi is inf 1 \n");
                    memcpy(sentence->xi,init_xi,sizeof(float)*num_keywords_neighbors);
                }else{
                    //printf("last_xi is not  nan or inf 1 \n");
                    memcpy(sentence->xi,last_xi,sizeof(float)*num_keywords_neighbors);
                }
            }
        }
        if ( has_neg_value_flag || last_z > (z = getXiFunction(sentence,model))) {
            learn_rate *= 0.5;
            z = last_z;
            eps = 10000;
            memcpy(sentence->xi,last_xi,sizeof(float)*num_keywords_neighbors);
        }
        else eps = util::norm2(last_xi,sentence->xi,num_keywords_neighbors);
        num_round ++;
    }
    while (num_round < max_xi_iter && eps > xi_min_eps);

    for (int i = 0; i < num_keywords_neighbors; i++) {
        if (isnan(-sentence->xi[i]) || isnan(sentence->xi[i]) || isinf(sentence->xi[i]) || isinf(-sentence->xi[i]) ){
            printf("doc->xi[i] nan here, so back \n");
            memcpy(sentence->xi, init_xi, sizeof(float) * num_keywords_neighbors);
            break;
        }
    }
    delete [] init_xi;
    delete [] last_xi;
    delete [] descent_xi;

}

void getDescentXi(Sentence* sentence, Model* model,float* descent_xi) {
    float sigma_xi = 0.0;
    float sigma_pi = 0.0;
    int num_keyword = sentence->num_keyword;
    int neighbor = model->neighbor;

    for (int i = 0; i < 2*neighbor +num_keyword; i++) {
        sigma_xi += sentence->xi[i];
    }

    for (int i = 0; i < num_keyword; i++) {
        sigma_pi += model->pi[sentence->keyword_ptr[i]];
    }

    for(int i = 0; i < 2*neighbor; i++){
        sigma_pi += model->pi[i+model->num_all_keywords];
    }


    for (int i = 0; i < num_keyword; i++) {
        descent_xi[i] = util::trigamma(sentence->xi[i]) * ( model->pi[sentence->keyword_ptr[i]] - sentence->xi[i]);
        descent_xi[i] -= util::trigamma(sigma_xi) * (sigma_pi - sigma_xi);
    }

    for (int i = 0; i < 2*neighbor; i++) {
        descent_xi[i+num_keyword] = util::trigamma(sentence->xi[i+num_keyword]) * (model->pi[i+model->num_all_keywords] - sentence->xi[i+num_keyword]);
        descent_xi[i+num_keyword] -= util::trigamma(sigma_xi) * (sigma_pi - sigma_xi);
    }

    int sen_num_words = sentence->num_words;

    int num_topics = model->num_topics;

    float* log_phi = model->log_phi;

    float* sum_log_phi = new float[num_topics];
    memset(sum_log_phi, 0.0, sizeof(float) * num_topics);

    for (int k = 0; k < num_topics; k++) {
        sum_log_phi[k] = 0.0;
        for (int i = 0; i < num_keyword; i++) {
            int keyword_id = sentence->keyword_ptr[i];
            sum_log_phi[k] += log_phi[keyword_id * num_topics + k] * sentence->xi[i];
        }
        for(int i = 0; i < 2*neighbor; i++){
            sum_log_phi[k] += sentence->log_neighbortopics[i*num_topics +k] * sentence->xi[i+num_keyword];
        }
    }


    float* sum_gamma_array = new float[num_topics];
    memset(sum_gamma_array, 0.0, sizeof(float) * num_topics);
    for (int k = 0; k < num_topics; k++) {
        sum_gamma_array[k] = 0.0;
        for (int i = 0; i < sen_num_words; i++) {
            sum_gamma_array[k] += exp(sentence->log_gamma[i * num_topics + k]) * sentence->words_cnt_ptr[i];
        }
    }

    float temp1 = 0;
    float temp2 = 0;
    for (int j = 0; j < num_keyword; j++) {
        for (int k = 0; k < num_topics; k++) {
            float sum_gamma = 0.0;
            temp1 += log_phi[sentence->keyword_ptr[j]* num_topics + k] * sigma_xi;
            sum_gamma = sum_gamma_array[k];
            temp1 -= sum_log_phi[k];
            temp1 = sum_gamma * (temp1/(sigma_xi * sigma_xi));
            descent_xi[j] += temp1;
        }
    }

    for (int j = 0; j < 2*neighbor; j++) {
        for (int k = 0; k < num_topics; k++) {
            float sum_gamma = 0.0;
            temp2 += sentence->log_neighbortopics[j*num_topics +k] * sigma_xi;
            sum_gamma = sum_gamma_array[k];
            temp2 -= sum_log_phi[k];
            temp2 = sum_gamma * (temp2/(sigma_xi * sigma_xi));
            descent_xi[j+num_keyword] += temp2;
        }
    }

    delete[] sum_log_phi;
    delete[] sum_gamma_array;
}


float getXiFunction(Sentence* sentence, Model* model) {
    float xi_function_value = 0.0;
    int neighbor = model->neighbor;
    int num_keyword = sentence->num_keyword;
    int num_all_keywords = model->num_all_keywords;

    int num_keywords_neighbors = sentence->num_keyword +2*neighbor;

    float sigma_xi = 0.0;
    float* pi = model->pi;
    float* log_phi = model->log_phi;

    // add all the xis.
    for (int i = 0; i < num_keywords_neighbors; i++) sigma_xi += sentence->xi[i];

    //consider the keywords' pi
    for (int i = 0; i < num_keyword; i++) {
        xi_function_value += (pi[sentence->keyword_ptr[i]] - sentence->xi[i])* (util::digamma(sentence->xi[i]) - util::digamma(sigma_xi)) + util::log_gamma(sentence->xi[i]);
    }

    //consider the neighbors' pi
    for(int i = 0; i < 2*neighbor; i++){
        xi_function_value += (pi[i + num_all_keywords] - sentence->xi[i + num_keyword]) * (util::digamma(sentence->xi[i + num_keyword]) - util::digamma(sigma_xi)) + util::log_gamma(sentence->xi[i + num_keyword]);
    }

    xi_function_value -= util::log_gamma(sigma_xi);

    int sen_num_words = sentence->num_words;
    int num_topics = model->num_topics;

    float* sum_log_theta = new float[num_topics];

    // consider the keywords
    for (int k = 0; k < num_topics; k++) {
        float temp = 0;
        for(int j = 0; j < num_keyword; j++){
            temp += log_phi[sentence->keyword_ptr[j] * num_topics + k] * sentence->xi[j]/sigma_xi;
        }
        sum_log_theta[k] = temp;
    }
    //consider the neighbors.
    for (int k = 0; k < num_topics; k++) {
        for(int i = 0; i < 2*neighbor; i++){
            sum_log_theta[k] += sentence->log_neighbortopics[i*num_topics +k] * sentence->xi[i+num_keyword]/sigma_xi;
        }
    }

    for (int i = 0; i < sen_num_words; i++) {
        for (int k = 0; k < num_topics; k++) {
            float temp = sum_log_theta[k];
            xi_function_value += temp * exp(sentence->log_gamma[i * num_topics + k]) * sentence->words_cnt_ptr[i];
        }
    }
    delete[] sum_log_theta;
    return xi_function_value;
}

inline void initXi(float* xi,int num) {
    for (int i = 0; i < num; i++) xi[i] = util::random();//init 100?!
}

void* ThreadInference(void* thread_data) {
    ThreadData* thread_data_ptr = (ThreadData*) thread_data;
    Document** corpus = thread_data_ptr->corpus;
    int start = thread_data_ptr->start;
    int end = thread_data_ptr->end;
    Configuration* configuration = thread_data_ptr->configuration;
    Model* model = thread_data_ptr->model;
    for (int i = start; i < end; i++) {
        doInference(corpus[i], model, configuration);
    }
    return NULL;
}

void runThreadInference(Document** corpus, Model* model, Configuration* configuration, int num_docs) {
    int num_threads = configuration->num_threads;
    pthread_t* pthread_ts = new pthread_t[num_threads];
    int num_per_threads = num_docs/num_threads;
    int i;
    ThreadData** thread_datas = new ThreadData* [num_threads];
    for (i = 0; i < num_threads - 1; i++) {
        thread_datas[i] = new ThreadData(corpus, i * num_per_threads, (i+1)*num_per_threads, configuration, model);;
        pthread_create(&pthread_ts[i], NULL, ThreadInference, (void*) thread_datas[i]);
    }
    thread_datas[i] = new ThreadData(corpus, i * num_per_threads, num_docs, configuration, model);
    pthread_create(&pthread_ts[i], NULL, ThreadInference, (void*) thread_datas[i]);
    for (i = 0; i < num_threads; i++) pthread_join(pthread_ts[i],NULL);
    for (i = 0; i < num_threads; i++) delete thread_datas[i];
    delete[] thread_datas;
}

inline void init_xi(float* xi,int num_labels) {
    for (int i = 0; i < num_labels; i++) xi[i] = util::random();//init 100?!
}

inline bool has_neg_value(float* vec,int dim) {
    for (int i =0; i < dim; i++) {
        if (vec[dim] < 0)return true;
    }
    return false;
}