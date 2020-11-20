from IPython.display import clear_output
from typing import List, Set, Union
from abc import abstractmethod
from functools import total_ordering
from os import path
import pickle
import os
import gc
import sys
import psutil
import time
import json
import os


class Index:
    def __init__(self):
        self.dic_index = {}
        self.set_documents = set()
        self.times_index = []
        self.init_time = time.time()
        self.end_time = time.time()

    def index(self, term: str, doc_id: int, term_freq: int):
        self.init_time = time.time()
        self.set_documents.add(doc_id)

        if term not in self.dic_index:
            int_term_id = len(self.dic_index) + 1
            self.dic_index[term] = self.create_index_entry(int_term_id)
        else:
            int_term_id = self.get_term_id(term)

        self.add_index_occur(self.dic_index[term], doc_id, int_term_id, term_freq)

    @property
    def vocabulary(self) -> List:
        return self.dic_index.keys()

    @property
    def document_count(self) -> int:
        return len(self.dic_index)

    @abstractmethod
    def get_term_id(self, term: str):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def create_index_entry(self, termo_id: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def add_index_occur(self, entry_dic_index, doc_id: int, term_id: int, freq_termo: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def get_occurrence_list(self, term: str) -> List:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def document_count_with_term(self, term: str) -> int:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def finish_indexing(self):
        pass

    def __str__(self):
        arr_index = []
        for str_term in self.vocabulary:
            arr_index.append(f"{str_term} -> {self.get_occurrence_list(str_term)}")

        return "\n".join(arr_index)

    def __repr__(self):
        return str(self)


@total_ordering
class TermOccurrence:
    def __init__(self, doc_id: int, term_id: int, term_freq: int):
        self.doc_id = doc_id
        self.term_id = term_id
        self.term_freq = term_freq

    def write(self, idx_file):
        pass

    def __hash__(self):
        return hash((self.doc_id, self.term_id, self.term_freq))

    def __eq__(self, other_occurrence: "TermOccurrence"):
        return ((self.doc_id, self.term_id, self.term_freq) ==
                (other_occurrence.doc_id, other_occurrence.term_id, other_occurrence.term_freq))

    def __lt__(self, other_occurrence: "TermOccurrence"):
        if not other_occurrence:
            return False
        return ((self.term_id, self.doc_id) <
                (other_occurrence.term_id, other_occurrence.doc_id))

    def __gt__(self, other_occurrence: "TermOccurrence"):
        if not other_occurrence:
            return False
        return ((self.term_id, self.doc_id) >
                (other_occurrence.term_id, other_occurrence.doc_id))

    def __str__(self):
        return f"(doc: {self.doc_id} term_id:{self.term_id} freq: {self.term_freq})"

    def __repr__(self):
        return str(self)


# HashIndex é subclasse de Index
class HashIndex(Index):
    def get_term_id(self, term: str):
        return self.dic_index[term][0].term_id

    def create_index_entry(self, term_id: int) -> List:
        return []

    def add_index_occur(self, entry_dic_index: List[TermOccurrence], doc_id: int, term_id: int, term_freq: int):
        entry_dic_index.append(TermOccurrence(doc_id, term_id, term_freq))

    def get_occurrence_list(self, term: str) -> List:
        return self.dic_index[term] if term in self.dic_index else []

    def document_count_with_term(self, term: str) -> int:
        return len(self.dic_index[term]) if term in self.dic_index else 0

    def finish_indexing(self):
        pass

    def calc_avg_time_mem(self):
        dic_convert = json.dumps(self.dic_index, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        f = open("dic_index.json", "w")
        f.write(dic_convert)
        f.close()

        with open('vocabulary.txt', 'a') as vocabulary_file:
            vocabulary_file.write('Vocabulary:\n')
            for key, value in self.dic_index.items():
                vocabulary_file.write(str(key) + "\n")


class TermFilePosition:
    def __init__(self, term_id: int, term_file_start_pos: int = None, doc_count_with_term: int = None):
        self.term_id = term_id
        self.term_file_start_pos = term_file_start_pos
        self.doc_count_with_term = doc_count_with_term

    def __str__(self):
        return f"term_id: {self.term_id}, doc_count_with_term: {self.doc_count_with_term}, term_file_start_pos: {self.term_file_start_pos}"

    def __repr__(self):
        return str(self)


class FileIndex(Index):
    TMP_OCCURRENCES_LIMIT = 1000000
    # TMP_OCCURRENCES_LIMIT = 10000

    def __init__(self):
        super().__init__()
        self.lst_occurrences_tmp = []
        self.idx_file_counter = 0
        self.str_idx_file_name = "occur_idx_file"

    def get_term_id(self, term: str):
        return self.dic_index[term].term_id

    def create_index_entry(self, term_id: int) -> TermFilePosition:
        return TermFilePosition(term_id)

    def add_index_occur(self, entry_dic_index: TermFilePosition, doc_id: int, term_id: int, term_freq: int):
        self.lst_occurrences_tmp.append(TermOccurrence(doc_id, term_id, term_freq))

        if len(self.lst_occurrences_tmp) >= FileIndex.TMP_OCCURRENCES_LIMIT:
            self.save_tmp_occurrences()

    def calc_avg_time_mem(self, total_time):
        print("Arquivos indexados: " + str(self.idx_file_counter))
        print("Tempo médio por arquivo: " + str(round(total_time / self.idx_file_counter, 2)) + "s")

        dic_convert = json.dumps(self.dic_index, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        f = open("dic_index.json", "w")
        f.write(dic_convert)
        f.close()

        with open('vocabulary.txt', 'a') as vocabulary_file:
            vocabulary_file.write('Vocabulary:\n')
            for key, value in self.dic_index.items():
                vocabulary_file.write(str(key) + "\n")



    def next_from_list(self) -> TermOccurrence:
        return self.lst_occurrences_tmp.pop(0) if self.lst_occurrences_tmp else None

    def next_from_file(self, file_idx: "BufferedReader") -> TermOccurrence:
        file_idx.read
        termOccurrence = None
        gc.disable()
        try:
            to = pickle.load(file_idx)

            if not to["doc_id"] and not to["term_id"] or not to["term_freq"]:
                raise Exception("Bad structure format")

            termOccurrence = TermOccurrence(to["doc_id"], to["term_id"], to["term_freq"])
        except:
            return None
        finally:
            gc.disable()
            return termOccurrence

    def save_tmp_occurrences(self):

        # ordena pelo term_id, doc_id
        # Para eficiencia, todo o codigo deve ser feito com o garbage
        # collector desabilitado
        gc.disable()

        # ordena pelo term_id, doc_id
        self.lst_occurrences_tmp = sorted(self.lst_occurrences_tmp)

        ### Abra um arquivo novo faça a ordenação externa: comparar sempre a primeira posição
        ### da lista com a primeira posição do arquivo usando os métodos next_from_list e next_from_file
        ### para armazenar no novo indice ordenado
        if self.idx_file_counter == 0:
            open(self.str_idx_file_name, "wb+").close()

        with open(self.str_idx_file_name, "rb") as r_file:

            self.str_idx_file_name = (
                (self.str_idx_file_name + str(self.idx_file_counter))
                if self.idx_file_counter == 0
                else self.str_idx_file_name.replace(str(self.idx_file_counter - 1), str(self.idx_file_counter)))

            with open(self.str_idx_file_name, "ab") as w_file:
                # RECURSIVE OPTION
                # def compare_and_add (nff, nfl):
                #     if not nff and not nfl:
                #         return None

                #     if (nff and not nfl) or nff < nfl:
                #         pickle.dump({ "doc_id": nff.doc_id, "term_id": nff.term_id, "term_freq": nff.term_freq}, w_file)
                #         compare_and_add(self.next_from_file(r_file), nfl)
                #     else:
                #         pickle.dump({ "doc_id": nfl.doc_id, "term_id": nfl.term_id, "term_freq": nfl.term_freq}, w_file)
                #         compare_and_add(nff, self.next_from_list())

                # compare_and_add(self.next_from_file(r_file), self.next_from_list())

                nff = self.next_from_file(r_file)
                nfl = self.next_from_list()
                while nff or nfl:
                    if (nff and not nfl) or nff < nfl:
                        pickle.dump({"doc_id": nff.doc_id, "term_id": nff.term_id, "term_freq": nff.term_freq}, w_file)
                        nff = self.next_from_file(r_file)
                    else:
                        pickle.dump({"doc_id": nfl.doc_id, "term_id": nfl.term_id, "term_freq": nfl.term_freq}, w_file)
                        nfl = self.next_from_list()

                self.idx_file_counter = self.idx_file_counter + 1

        gc.enable()

    def finish_indexing(self):
        if len(self.lst_occurrences_tmp) > 0:
            self.save_tmp_occurrences()

        # Sugestão: faça a navegação e obetenha um mapeamento
        # id_termo -> obj_termo armazene-o em dic_ids_por_termo

        dic_ids_por_termo = {}
        for str_term, obj_term in self.dic_index.items():
            dic_ids_por_termo[obj_term.term_id] = {"term": str_term, "tfp": obj_term}

        # navega nas ocorrencias para atualizar cada termo em dic_ids_por_termo
        # apropriadamente

        size = os.path.getsize(self.str_idx_file_name)
        if size < 8589934592:
            with open(self.str_idx_file_name, 'rb') as idx_file:
                pos = 0
                while True:
                    to = self.next_from_file(idx_file)
                    if not to:
                        break

                    if to.term_id in dic_ids_por_termo:
                        dic_ids_por_termo[to.term_id]["tfp"].doc_count_with_term = (
                                self.document_count_with_term(dic_ids_por_termo[to.term_id]["term"]) + 1)
                        dic_ids_por_termo[to.term_id]["tfp"].term_file_start_pos = pos
                    pos = pos + 1
        else:
            print("Arquivo indexado maior que 8GB. Memória insificiente")

        for i, e in dic_ids_por_termo.items():
            self.dic_index[e["term"]] = e["tfp"]

    def get_occurrence_list(self, term: str) -> List:
        occurrence_list = []
        if not term in self.dic_index:
            return []

        with open(self.str_idx_file_name, 'rb') as idx_file:
            while True:
                to = self.next_from_file(idx_file)
                if not to:
                    break
                else:
                    if to.term_id == self.dic_index[term].term_id:
                        occurrence_list.append(to)

        return occurrence_list

    def document_count_with_term(self, term: str) -> int:
        return (
            self.dic_index[term].doc_count_with_term
            if term in self.dic_index and self.dic_index[term].doc_count_with_term
            else 0)


class Indexer:
    # array de palavras
    vocabulary = []
    # array de ocorrencias por palavra onde cada posicao corresponde a posicao do vocabulary
    occurence = []
    # chave e um inteiro e o valor da chave corresponde a uma tupla de palavra de ocorrencia
    indextable = {}

    def _init_(self, document):
        self.document = document
        self.vocabulary = None
        self.occurence = None
        self.indextable = None

    # adiciona uma palavra a index table
    def add_word(self, palavra):
        pass

    # adicionar ocorrência de uma palavra
    def add_ocorrence(self, palavra):
        pass

    # percorre o documento
    def paser_doc(self):
        pass
