from gensim import corpora, models, similarities, matutils

model_path = '/DATA/lyl/model/'
lsi_topic_num = 500

class Model:
    def __init__(self, texts, save_id = None):
        print(save_id)
        if save_id is not None:
            try:
                self.dictionary = corpora.Dictionary.load(model_path + '%s.dictionary' % save_id)
                self.tfidf = models.TfidfModel.load(model_path + '%s.tfidf' % save_id)
                self.lsi = models.LsiModel.load(model_path + '%s_%s.lsi' % (save_id, lsi_topic_num))
                print('model already exists!')
                return
            except:

                pass
        
        if (texts is None) or (texts == []):
            raise Exception('error on init nlp Model')
            
        self.dictionary = corpora.Dictionary(texts)
        
        corpus = [self.dictionary.doc2bow(text) for text in texts]     
        
        self.tfidf = models.TfidfModel(corpus)
        
        corpus_tfidf = self.tfidf[corpus]
                
        self.lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=lsi_topic_num)
        
        # save model
        if save_id is not None:
            print('save model: ', save_id)
            self.dictionary.save(model_path + '%s.dictionary' % save_id)
            self.tfidf.save(model_path + '%s.tfidf' % save_id)
            self.lsi.save(model_path + '%s_%s.lsi' % (save_id, lsi_topic_num))

    def get_tfidf(self, tokens):
        query_bow = self.dictionary.doc2bow(tokens)
        query_tfidf = self.tfidf[query_bow]
        return query_tfidf
    
    def get_lsi(self, tokens):
        query_bow = self.dictionary.doc2bow(tokens)
        query_tfidf = self.tfidf[query_bow]
        query_lsi = self.lsi[query_tfidf]
        return query_lsi
    
    def query_sim_tfidf(self, tokens1, tokens2):
        return matutils.cossim(self.get_tfidf(tokens1), self.get_tfidf(tokens2))
    
    def query_sim_lsi(self, tokens1, tokens2):
        return matutils.cossim(self.get_lsi(tokens1), self.get_lsi(tokens2))

if __name__ == "__main__":
    documents = ["Shipment of gold damaged in a fire", "Delivery of silver arrived in a silver truck", "Shipment of gold arrived in a truck", "orz"]
    texts = [[word for word in document.lower().split()] for document in documents]
    m = Model(texts)
    z1 = ['water', 'gold',  'in', 'the', 'shipment', 'shipment']
    z2 = ['aaa', 'bbb', 'a', 'gold', 'in', 'fire', 'in']

    print(m.query_sim_tfidf(z1, z2))
    print(m.query_sim_lsi(z1, z2))
    
    print(m.query_sim_tfidf(['gold', 'in', 'shipment', 'shipment', 'orz'],['shipment', 'in', 'fire']))
    print(m.query_sim_lsi(['gold', 'in', 'shipment', 'shipment', 'orz'],['shipment', 'in', 'fire']))
    
