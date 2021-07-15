import time
from pathlib import Path

def get_train_data(data_dir, min_text_length, printout):
    ''' Collecting texts from data_dir.
        If data_dir is ../CLEAN_DATA than no need to save or clean data.'''
        
    t0 = time.time()
    if data_dir is not None:
        if printout == True:
            print('Started colecting texts from {}....'.format(data_dir))
    
        all_texts = []
        all_cats = []
        categories = ['comp.sys.ibm.pc.hardware',
                      'rec.sport.hockey',
                      'talk.politics.guns',]

        for sub_dir in Path(data_dir).iterdir():
            if sub_dir.is_dir() and sub_dir.name in categories:
                new_subdir = Path(sub_dir.parent, 'CLEAN_DATA', sub_dir.name)
                if new_subdir.exists:
                    save_data = False
                    clean_data = False
                else:
                    save_data = True
                    new_subdir.mkdir(parents=True, exist_ok=True)
                texts = []
                for filename in sub_dir.iterdir():
                    with open(filename, 'rb') as f:
                        text = f.read()

                    if len(text) > min_text_length:
                        texts.append(str(text))
                        
                if clean_data == "True":
                    texts = clean_train_data(texts, printout)
                    
                if save_data == True:
                    for i, text in enumerate(texts):
                         new_filename = Path(new_subdir, "file_{}".format(i) + '.txt')
                         new_filename.write_text(text)
                         
                all_texts += texts
                all_cats += [str(sub_dir.name)] * len(texts)

    if printout == True:
        print('Texts collecion took {}'.format(int(time.time() - t0))) 
        
    return all_texts, all_cats

def clean_train_data(all_texts, printout):
    '''Using Spacy NLP lib to remove all OOV and stop words, punctuation, and pronouns'''
    
    if printout == True:
        print('Started cleaning texts...')
    import spacy
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "textcat"])
    docs = list(nlp.pipe(all_texts))
    all_cln_texts = []
    for doc in docs:
        cln_text = [tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_oov and not tok.is_stop and tok.pos_ != 'PRON']
        all_cln_texts.append(' '.join(cln_text))
    return all_cln_texts