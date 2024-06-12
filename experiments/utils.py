import re
import json
import string
from datetime import datetime
from pathlib import Path
import nmslib
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import os
# from IPython.core.debugger import set_trace
from bs4 import BeautifulSoup
from cleantext.sklearn import CleanTransformer
# from lingua import Language, LanguageDetectorBuilder
# from setfit import SetFitModel
import numpy as np
import ast
from stop_words import get_stop_words
from collections import Counter
import nltk
from HanTa import HanoverTagger as ht
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ranx import evaluate
from sentence_transformers import SentenceTransformer, util, CrossEncoder

#-------------------- Preprocessing ------------------

def custom_stopwords(documents, threshold=0.5):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from collections import defaultdict

    # Ensure necessary resources are downloaded
    """
    Calculate custom stopwords from a list of documents based on frequency.
    
    Args:
    - documents (list): A list of text documents.
    - threshold (float): The fraction (0 to 1) threshold to decide stopwords. E.g., 0.5 means a word is a stopword if it appears in more than 50% of documents.
    
    Returns:
    - set: A set of custom stopwords.
    """
    
    num_docs = len(documents)
    word_doc_freq = defaultdict(int)
    
    for doc in documents:
        tokens = set(word_tokenize(doc.lower()))  # Convert document to unique set of words
        for token in tokens:
            word_doc_freq[token] += 1
    
    # Words that appear in more than threshold fraction of documents are considered stopwords
    custom_stops = {word for word, freq in word_doc_freq.items() if freq / num_docs > threshold}
    
    # Combine with NLTK's default stopwords (optional)
    # combined_stops = custom_stops.union(set(stopwords.words('german')))
    
    return custom_stops


def preprocess_documents(text, lemmatization_model):
    text = get_lemmas(remove_stop_words(clean_text(remove_html_elements(text)), 'de'), lemmatization_model)
    # print(text)
    return text

def remove_enclosed_content(s):
    return re.sub(r'<[^>]*>', '', s)

def get_lemmas(text, model):
    words = nltk.word_tokenize(text)
    lemmata = [e[1] for e in model.tag_sent(words)]
    return ' '.join(lemmata).lower()


def get_stems(text, model):
    """
    Lemmatize German words in the given text.
    
    Args:
    - text (str): The input German text.
    
    Returns:
    - str: Lemmatized text.
    """    
    words = nltk.word_tokenize(text)
    lemmas = [e[1] for e in model.tag_sent(words)]
    return ' '.join(lemmas)


def remove_stop_words(text, lang):
    stop_words = stop_words = get_stop_words(lang) + ["antwort", "wählen","folgenden","richtige", "bitte", "aufgabe", "aufgaben", "z", "b", "etc", "verwendet", "aussagen", "lena", "nächst", "thema", "rating", "nummer", "true", "antworttext", "richtig", "bild", "false", "=", ">", "<", "rating", "antwort", "text", "wort", "falsch", "neu", "na"]
    stopwords_dict = Counter(stop_words)
    text = ' '.join([word for word in text.split() if word not in stopwords_dict])
    return text

def remove_html_elements(text):
    try:
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        return text.replace('z.B.','').replace('&nbsp;','').replace("<u>", "").replace(">>", "").replace("<u>nicht<u>", "nicht").replace(">>", "")
    except TypeError:
        return ''

def clean_text(document):

    cleaner = CleanTransformer(fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=True,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=True,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="nummer",
        replace_with_currency_symbol="<CUR>",
        lang="de")
    result = cleanhtml(cleaner.transform([document]))
    return remove_enclosed_content(result[0])

def detect_lang(text):
    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN,\
                 Language.SPANISH, Language.CZECH, Language.ARABIC, Language.PORTUGUESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    return str(detector.detect_language_of(text)).split('.')[1]

def extract_correct_answers(row):
    for e in row:
        if e['column_value']=='1':
            return e['text'].lower()
        else:
            pass

def preprocess_json_data(json_path):
    df = pd.read_json(json_path)
    df['points'] = df['points'].fillna('N/A')
    df['questions_temp'] = df['text'].apply(remove_html_elements)
    df['questions_temp'] = df['questions_temp'].apply(lambda x: x.replace('____', 'spacecompletion'))
    df['questions_temp'] = df['questions_temp'].apply(clean_text)

    df['correct_answers_temp'] = df['answers'].apply(extract_correct_answers)
    df['correct_answers_temp'] = df['correct_answers_temp'].apply(remove_html_elements)
    df['correct_answers_temp'] = df['correct_answers_temp'].apply(clean_text)

    df['frontend_info_text'] = df['frontend_info_text'].apply(remove_html_elements)
    df['frontend_info_text'] = df['frontend_info_text'].apply(clean_text)
    
    df.rename(columns={'test_title': 'onlinetest_title'}, inplace=True)
    df['onlinetest_title'] = df['onlinetest_title'].apply(clean_text)
    df['content'] = df['questions_temp'] + ' ' + df['correct_answers_temp'] #+ ' ' + df['onlinetest_title']
    df['lang'] = df['content'].apply(detect_lang)
    print(f"Number of items before removing duplicates: {df.shape[0]}")
    df = df.drop_duplicates(subset=['content'], keep='first')
    df.reset_index(inplace=True, drop=True)
    df['docid'] = df.index
    return df

def translate(sentence):
    """
    Translate from German to English
    """
    translator = Translator(to_lang="en", from_lang='de', provider='libre')
    return translation


# -----------------------------------------



# ---------------- Other Functionalities -------------




def save_dict_as_json(a_dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        return json.dump(a_dict, f, ensure_ascii=False, sort_keys=True, indent=4, default=default)


def read_json(path):
    """
    Reads JSON file/data

    Parameters
    ----------
    path: string
        The JSON file path
    Returns
    -------
    data: string
        The dictionary

    """
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def save_to_json(a_dict, path, **kwargs):
    """
    Save a dict as a json
    """
    path = list(Path(path).parts)
    # now = datetime.now()
    # date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    if 'model_name' in kwargs:
        model_name = kwargs['model_name']
    else:
        model_name = ''
    path = Path('/'+'/'.join(path[1:-1])+'/'+model_name+'_'+path[-1])
    Path(path).parent.mkdir(parents=True, exist_ok=True)  
#     print(path, type(path))
    with open(str(path), 'w', encoding='utf-8') as f:
        json.dump(a_dict, f, ensure_ascii=False, indent=4)
        

# def create_index(documents_set, model, **kwargs):
#     '''
#     Create index
    
#     documents_set: A dataframe with all the context to be indexed.
#     model: A language model to calculate embeddings.
#     index_path: The path for storing the index in case one needs to store it.
#     '''
#     # print("size:", len(documents_set['content'].tolist()))
#     index = nmslib.init(method='hnsw', space='cosinesimil')
#     # model.tokenizer.pad_token = model.tokenizer.eos_token
#     # model.tokenizer.mask_token = '[MASK]' # or another appropriate token
#     if model.tokenizer.pad_token is None:
#     # Check if the tokenizer has a default pad token
#         if hasattr(model.tokenizer, 'pad_token_id'):
#             model.tokenizer.pad_token = model.tokenizer.pad_token_id
#         else:
#             # Set a standard pad token if the tokenizer supports it
#             # This part might need adjustment based on the specific tokenizer
#             model.tokenizer.pad_token = model.tokenizer.eos_token
#     try:
#         document_set_embeddings = [model.encode(e) for e in documents_set['content'].tolist()]
#     except ValueError:
#         model.tokenizer.pad_token = model.tokenizer.eos_token
#         model.tokenizer.mask_token = '[MASK]' # or another appropriate token
#         document_set_embeddings = [model.encode(e) for e in documents_set['content'].tolist()]
        
    
    
#     index.addDataPointBatch(document_set_embeddings)
#     index.createIndex({'post': 2})

#     if 'index_path' in kwargs:
#         index.saveIndex(kwargs['index_path'])
#     else:
#         pass
#     return index

# def create_index(documents_set, model, **kwargs):
#     '''
#     Create index
    
#     documents_set: A dataframe with all the context to be indexed.
#     model: A language model to calculate embeddings.
#     index_path: The path for storing the index in case one needs to store it.
#     '''
#     # print("size:", len(documents_set['content'].tolist()))
#     index = nmslib.init(method='hnsw', space='cosinesimil')
#     document_set_embeddings = [model.encode(e) for e in documents_set['content'].tolist()]
#     index.addDataPointBatch(document_set_embeddings)
#     index.createIndex({'post': 2, 'num_threads':11})

#     if 'index_path' in kwargs:
#         index.saveIndex(kwargs['index_path'])
#     else:
#         pass
#     return index




def create_index(documents_set, model, **kwargs):
    '''
    Create index
    
    documents_set: A dataframe with all the context to be indexed.
    model: A language model from the Sentence Transformers library to calculate embeddings.
    index_path: The path for storing the index in case one needs to store it.
    '''
    index = nmslib.init(method='hnsw', space='cosinesimil')

    # Attempt to get a specific name of the model
    model_name = getattr(model, 'name', getattr(model, 'model_name', 'Unknown Model'))

    # Progress bar for encoding documents, including model name
    encoding_description = f"Encoding Documents with {model_name}"
    document_set_embeddings = [model.encode(e) for e in tqdm(documents_set['content'].tolist(), desc=encoding_description)]

    index.addDataPointBatch(document_set_embeddings)

    # Simulate progress for index creation, including model name
    index_creation_description = f"Creating Index with {model_name}"
    with tqdm(total=2, desc=index_creation_description) as pbar:
        index.createIndex({'M': 200, 'efConstruction': 200, 'post': 2, 'searchMethod':200})
        pbar.update(1)  # Update progress after index creation

        if 'index_path' in kwargs:
            index.saveIndex(kwargs['index_path'])
            pbar.update(1)  # Update progress after index saving

    return index


def to_dic(e):
    try:
        return ast.literal_eval(e)
    except (ValueError, SyntaxError):
        return {"answers":e}


def re_rank_dataframe(df, query, model):
    query_doc_pairs = [(query, doc) for doc in df['content']]
    scores = model.predict(query_doc_pairs)
    if len(scores.shape) == 2 and scores.shape[1] == 2:
        scores = scores[:, 0]
    df['score'] = scores
    df_sorted = df.sort_values(by='score', ascending=True)
    return df_sorted


re_ranking_model = CrossEncoder("amberoad/bert-multilingual-passage-reranking-msmarco")


def search(query, index, model, df, **kwargs):
    """
    Approximate nearest neighbor search
    
    Parameters
    ----------
    query: A string query
    index: An NMSLIB index
    model: A language model
    df: The content against the query will be runned
    run_path: the path where the results will be stored (in JSON format)
    k: The number of elements to be indexed by the search, by default is 10.
    
    Returns
    -------
    """
    query_embeddings = model.encode(query)
    model_name = Path(str(model.tokenizer).split()[0].split("=")[1].replace("'","").replace(",","")).parts[-1:][0]
    # With the index, make a query and approximate its 25 nearest neighbors
    if 'k' in kwargs:
        k = kwargs['k']
    else:
        # k = df.shape[0]
        k = 100
    ids, distances = index.knnQuery(query_embeddings, k=k)
    # Process the output
    indices_and_weights = list(zip(ids, distances))
    
    nmslib_indices = [e[0] for e in indices_and_weights]
    results_df = df.iloc[nmslib_indices]
        
    retrieved_question_ids = [e for e in results_df['docid'].tolist()]
    # print(retrieved_question_ids)
    # TODO: check if it is necessary to reset the indices in the DF,
    # in the original DF there were repeated elements in the content column
    # for consistency with the original data, the new reference id will be removed
    results = df.query("docid in @retrieved_question_ids")[:k]#.reset_index(drop=True)[:k]
    # print(results['answers'].tolist())
    # results['query_embedding'] = [query_embeddings]*results.shape[0] # this is for plotting, still needs to be reduced

    results = re_rank_dataframe(results, query, re_ranking_model)

    results = results[['docid', 'onlinetest_title', 'question_type_id', 'question_type_name', 
                       'answer_type_id', 'answer_type_name', 'text', 'correct_answers_temp','variable', 'points', 'answers', 'source', 'level_difficulty', 'topic_label_de_fixed', 'related_topics']]

    

    results['answers'] = results['answers'].apply(to_dic)
    results['related_topics'] = results['related_topics'].apply(to_dic)

    
    results['points'] = results['points'].fillna('N/A')
    results['onlinetest_title'] = results['onlinetest_title'].fillna('N/A')
    results['question_type_id'] = results['question_type_id'].fillna('N/A')
    results['question_type_name'] = results['question_type_name'].fillna('N/A')
    results['variable'] = results['variable'].fillna('N/A')
    results['correct_answers_temp'] = results['correct_answers_temp'].fillna('N/A')

    
    topic_filters = list(set(results['topic_label_de_fixed'].tolist()))
    answer_type_filters = list(set(results['answer_type_name'].tolist()))
    question_type_filters = list(set(results['question_type_name'].tolist()))
    question_difficulty_filters = list(set(results['level_difficulty'].tolist()))
    source_filters = list(set(results['source'].tolist()))
    points_filters = list(set(results['points'].tolist()))

    results = [e[1] for e in results.T.to_dict().items()]
    
    for i, j in zip(results, indices_and_weights):
        i['cosine_distance'] = str(j[1])
    for e in results:
        e['query']=str(query)

    filter_data = {
        'topic_filters':topic_filters, 
        'answer_type_filters': answer_type_filters,
        'question_type_filters': question_type_filters,
        'question_difficulty_filters': question_difficulty_filters,
        'source_filters':source_filters, 
        'points_filters':points_filters
        }
    

    results = {
        'filter_data' : filter_data,
        'search_output' : results
        }

    if 'run_path' in kwargs:
        save_to_json([dict(e) for e in results['search_output']], kwargs['run_path'], model_name=model_name) # corregir este parseo ya no aplica!!!
    else:
        pass
    # For debugging the output
    # save_dict_as_json(results, "/Users/user/Downloads/newwweeww_outputtttt.json")
    return results


def load_nmslib_index(path):
    """
    Loads NMSLIB index
    """
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(path)
    return index
    



def get_cv_metrics(X, y, model, cv_strategy):
    from sklearn.metrics import  balanced_accuracy_score, precision_score, recall_score
    import numpy as np

    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    cnt=0
    for train_index, test_index in cv_strategy.split(X, y):
        cnt = cnt + 1
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        model.fit(x_train_fold, y_train_fold, num_epochs=10)
        predictions = model.predict(x_test_fold)
        # predictions_probas = clf.predict_proba(x_test_fold)

        print("----------------")
        print("--- Info ---")
        print(f"--> Fold {cnt}")    
        print("--> Ground Truth (y_test_fold):", y_test_fold)    
        print("--> Predictions:               ", predictions)
        
            
        acc = balanced_accuracy_score(y_test_fold, predictions, adjusted=True)
        prec = precision_score(y_test_fold, predictions, average='weighted', zero_division=True)
        rec = recall_score(y_test_fold, predictions, average='weighted', zero_division=True)
        
        print("\n--- Metrics ---")
        print(f"-> Accuracy: {acc}")
        print(f"-> Precision: {prec}")
        print(f"-> Recall: {rec}\n")
        
        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)

        
    print("\n-------------------AVG Metrics-------------------------")
    print("-> AVG Accuracy:", np.mean(accuracy_scores))
    print("-> AVG Precision:", np.mean(precision_scores))
    print("-> AVG Recall:", np.mean(recall_scores))

    return {'accuracy':accuracy_scores, 'precision': precision_scores, 'recall':recall_scores}


def list_files_from_dir(path, **kwargs):
    """
    list files from directory
    path: the path of the directory
    extention: filter by extention
    """
    list_of_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'extention' in kwargs:
                if file.endswith(kwargs['extention']):
                    list_of_files.append(os.path.join(root, file))
            else:
                list_of_files.append(os.path.join(root, file))
    return list_of_files


def cleanhtml(raw_html):
    try:
        raw_html = re.sub(r'<br\s?\/>|<br>', "", raw_html)
        raw_html = re.sub("</?p>","",raw_html)
        cleantext = BeautifulSoup(raw_html, "lxml").text
        return ''.join(cleantext).translate(str.maketrans('', '', string.punctuation))
    except TypeError:
        return raw_html



def to_trec_format(a_path):
    if Path(a_path).suffix == '.json':
        results = read_json(a_path)
        ids = []
        query = []
        similarity = []
        for i,j in enumerate(results):
            ids.append(j['docid'])
            query.append(j['query'])
            try:
                similarity.append(j['cosine_distance'])
            except KeyError:
                similarity.append(j['BM25_score'])
        a_df = pd.DataFrame({'docno':ids, 'qid':query,'score':similarity})
        #         model_name = str('-'.join(Path(e).parts[-1:][0].split('_')[2:-2]))
        model_name = list(Path(a_path).parts)[-1:][0]
        model_name = [e for e in model_name.split('.') if e!='json']
        model_name = ['_'.join(e.split("_")[:-1][:-1]) for e in model_name][0]
        

        a_df['tag'] = model_name
        a_df['Q0'] = 'Q0'
        a_df['rank'] = a_df.index+1
        a_df = a_df[['qid', 'Q0', 'docno', 'rank', 'score', 'tag']]
        
        tsv_path = Path(a_path).with_suffix('.tsv')
        return a_df.to_csv(tsv_path, index = False, sep = '\t', header=False)
    else:
        pass


def extract_keywords(text, n, model):
    kw_model = KeyBERT(model)
    return [e[0] for e in kw_model.extract_keywords(text)][:n]


def assign_difficulty_label(value):    
    if 0.0 <= value <= 0.50:
        return "easy"
    elif 0.51 <= value <= 0.70:
        return "medium"
    elif 0.71 <= value <= 1.0:
        return "hard"
    else:
        return "Unknown"


#--------- Predicting Related Topics via-FSL
# Usage
# get_related_topics("welches ist die beliebteste", mappings, intent_model_path)

def load_model(utterance, path):
    saved_model = SetFitModel._from_pretrained(path)
    return saved_model.predict([utterance]).cpu().detach().numpy()



def encode_utterance(utterance, model):
    return model.encode(utterance, convert_to_tensor=False)

def get_cosine_sim(utterance, q_type, model):
    return util.cos_sim(encode_utterance(utterance, model),encode_utterance(q_type, model)).numpy()[0][0]


def get_related_topics(doc, mappings, intent_model_path):
    saved_model = SetFitModel._from_pretrained(intent_model_path)
    mappings = read_json(mappings)
    probas = list(saved_model.predict_proba([doc]).cpu().detach().numpy()[0])
    mappings = [e for e in mappings.keys()]
    result = [{'topic':i, 'rate':j} for i , j in zip(mappings, probas)]
    return result



def get_llama_emb(list_of_docs):
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    t = AutoTokenizer.from_pretrained(model_id)
    t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", use_auth_token='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' )
    m.eval()

    t_input = t(list_of_docs, padding=True, return_tensors="pt")

    with torch.no_grad():
        last_hidden_state = m(**t_input, output_hidden_states=True).hidden_states[-1]

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

    return sentence_embeddings.detach().cpu().numpy()



def get_last_element_of_path(path):
    return os.path.basename(path).split('_run.txt')[0]



def create_report(qrels, runs, runs_names, k, path=None):
    results = []
    for i in runs:
        metrics = pd.DataFrame([evaluate(qrels, i, [f"precision@{k}",f'recall@{k}',
                                                    f'f1@{k}', f'mrr@{k}', f'ndcg@{k}', f'map@{k}'])])
        results.append(metrics)
    # print(results)
    results = pd.concat(results).reset_index(drop=True)
    results['models'] = runs_names
    results = results[["models",  f'ndcg@{k}', f'mrr@{k}', 
                       f"precision@{k}",f'recall@{k}', f'f1@{k}', f'map@{k}']]

    if path is not None:
        results.to_csv(path, index=False)

    return results
