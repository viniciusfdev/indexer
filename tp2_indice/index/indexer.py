from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import string
import os

class Cleaner:
    def __init__(self,stop_words_file:str,language:str,
                        perform_stop_words_removal:bool,perform_accents_removal:bool,
                        perform_stemming:bool):
        self.set_stop_words = self.read_stop_words(stop_words_file)

        self.stemmer = SnowballStemmer(language)
        in_table =  "áéíóúâêôçãẽõü"
        out_table = "aeiouaeocaeou"
        
        self.accents_translation_table = str.maketrans(in_table, out_table)
        self.set_punctuation = set(string.punctuation)

        #flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self,html_doc:str) ->str:
        try: 
            return BeautifulSoup(html_doc, features="lxml").get_text()
        except: 
            return None

    def read_stop_words(self,str_file):
        set_stop_words = set()
        with open(str_file, "r") as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split(",")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words
    
    def is_stop_word(self,term: str):
        return True if term in self.set_stop_words else False

    def word_stem(self,term:str):
        return self.stemmer.stem(term)


    def remove_accents(self,term: str) -> str:
        return term.translate(self.accents_translation_table)

    def is_accent(self, term: str) -> str:
        return self.remove_accents(term) == term

    def preprocess_word(self,term:str) -> str:
        if self.perform_stop_words_removal and self.is_stop_word(term) and self.is_accent(term):
            return None
        
        term = str.lower(term)

        if self.perform_accents_removal:
            term = self.remove_accents(term)  

        if self.perform_stemming:
            term = self.word_stem(term)

        return term

class HTMLIndexer:
    cleaner = Cleaner(stop_words_file="stopwords.txt",
                        language="portuguese",
                        perform_stop_words_removal=True,
                        perform_accents_removal=True,
                        perform_stemming=True)
    def __init__(self,index):
        self.index = index

    def text_word_count(self,plain_text:str):
        dic_word_count = {}

        return dic_word_count
    def index_text(self,doc_id:int, text_html:str):
        pass

    def index_text_dir(self,path:str):

        for str_sub_dir in os.listdir(path):
            path_sub_dir = f"{path}/{str_sub_dir}"
