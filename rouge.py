# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import six
import io
import os
from copy import deepcopy
import itertools


class FilesRouge:
    def __init__(self, *args, **kwargs):
        """See the `Rouge` class for args
        """
        self.rouge = Rouge(*args, **kwargs)
    
    def _check_files(self, hyp_path, ref_path):
        assert (os.path.isfile(hyp_path))
        assert (os.path.isfile(ref_path))
        
        def line_count(path):
            count = 0
            with open(path, "rb") as f:
                for line in f:
                    count += 1
            return count
        
        hyp_lc = line_count(hyp_path)
        ref_lc = line_count(ref_path)
        assert (hyp_lc == ref_lc)
    
    def get_scores(self, hyp_path, ref_path, avg = False, ignore_empty = False):
        """Calculate ROUGE scores between each pair of
        lines (hyp_file[i], ref_file[i]).
        Args:
          * hyp_path: hypothesis file path
          * ref_path: references file path
          * avg (False): whether to get an average scores or a list
        """
        self._check_files(hyp_path, ref_path)
        
        with io.open(hyp_path, encoding = "utf-8", mode = "r") as hyp_file:
            hyps = [line[:-1] for line in hyp_file]
        
        with io.open(ref_path, encoding = "utf-8", mode = "r") as ref_file:
            refs = [line[:-1] for line in ref_file]
        
        return self.rouge.get_scores(hyps, refs, avg = avg,
                                     ignore_empty = ignore_empty)


class Rouge:
    DEFAULT_METRICS = ["rouge-1", "rouge-2", "rouge-l"]
    AVAILABLE_METRICS = {
        "rouge-1": lambda hyp, ref, **k: rouge_n(hyp, ref, 1, **k),
        "rouge-2": lambda hyp, ref, **k: rouge_n(hyp, ref, 2, **k),
        "rouge-3": lambda hyp, ref, **k: rouge_n(hyp, ref, 3, **k),
        "rouge-4": lambda hyp, ref, **k: rouge_n(hyp, ref, 4, **k),
        "rouge-5": lambda hyp, ref, **k: rouge_n(hyp, ref, 5, **k),
        "rouge-l": lambda hyp, ref, **k:
        rouge_l_summary_level(hyp, ref, **k),
    }
    DEFAULT_STATS = ["r", "p", "f"]
    AVAILABLE_STATS = ["r", "p", "f"]
    
    def __init__(self, metrics = None, stats = None, return_lengths = False,
                 raw_results = False, exclusive = True):
        self.return_lengths = return_lengths
        self.raw_results = raw_results
        self.exclusive = exclusive
        
        if metrics is not None:
            self.metrics = [m.lower() for m in metrics]
            
            for m in self.metrics:
                if m not in Rouge.AVAILABLE_METRICS:
                    raise ValueError("Unknown metric '%s'" % m)
        else:
            self.metrics = Rouge.DEFAULT_METRICS
        
        if self.raw_results:
            self.stats = ["hyp", "ref", "overlap"]
        else:
            if stats is not None:
                self.stats = [s.lower() for s in stats]
                
                for s in self.stats:
                    if s not in Rouge.AVAILABLE_STATS:
                        raise ValueError("Unknown stat '%s'" % s)
            else:
                self.stats = Rouge.DEFAULT_STATS
    
    def get_scores(self, hyps, refs, avg = False, ignore_empty = False):
        if isinstance(hyps, six.string_types):
            hyps, refs = [hyps], [refs]
        
        if ignore_empty:
            # Filter out hyps of 0 length
            hyps_and_refs = zip(hyps, refs)
            hyps_and_refs = [_ for _ in hyps_and_refs
                             if len(_[0]) > 0
                             and len(_[1]) > 0]
            hyps, refs = zip(*hyps_and_refs)
        
        assert (isinstance(hyps, type(refs)))
        assert (len(hyps) == len(refs))
        
        if not avg:
            return self._get_scores(hyps, refs)
        return self._get_avg_scores(hyps, refs)
    
    def _get_scores(self, hyps, refs):
        scores = []
        for hyp, ref in zip(hyps, refs):
            sen_score = {}
            
            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]
            
            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(
                    hyp,
                    ref,
                    raw_results = self.raw_results,
                    exclusive = self.exclusive)
                sen_score[m] = {s: sc[s] for s in self.stats}
            
            if self.return_lengths:
                lengths = {
                    "hyp": len(" ".join(hyp).split()),
                    "ref": len(" ".join(ref).split())
                }
                sen_score["lengths"] = lengths
            scores.append(sen_score)
        return scores
    
    def _get_avg_scores(self, hyps, refs):
        scores = {m: {s: 0 for s in self.stats} for m in self.metrics}
        if self.return_lengths:
            scores["lengths"] = {"hyp": 0, "ref": 0}
        
        count = 0
        for (hyp, ref) in zip(hyps, refs):
            hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]
            
            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref, exclusive = self.exclusive)
                scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}
            
            if self.return_lengths:
                scores["lengths"]["hyp"] += len(" ".join(hyp).split())
                scores["lengths"]["ref"] += len(" ".join(ref).split())
            
            count += 1
        avg_scores = {
            m: {s: scores[m][s] / count for s in self.stats}
            for m in self.metrics
        }
        
        if self.return_lengths:
            avg_scores["lengths"] = {
                k: scores["lengths"][k] / count
                for k in ["hyp", "ref"]
            }
        
        return avg_scores


class Ngrams(object):
    """
        Ngrams datastructure based on `set` or `list`
        depending in `exclusive`
    """
    
    def __init__(self, ngrams = {}, exclusive = True):
        if exclusive:
            self._ngrams = set(ngrams)
        else:
            self._ngrams = list(ngrams)
        self.exclusive = exclusive
    
    def add(self, o):
        if self.exclusive:
            self._ngrams.add(o)
        else:
            self._ngrams.append(o)
    
    def __len__(self):
        return len(self._ngrams)
    
    def intersection(self, o):
        if self.exclusive:
            inter_set = self._ngrams.intersection(o._ngrams)
            return Ngrams(inter_set, exclusive = True)
        else:
            other_list = deepcopy(o._ngrams)
            inter_list = []
            
            for e in self._ngrams:
                try:
                    i = other_list.index(e)
                except ValueError:
                    continue
                other_list.pop(i)
                inter_list.append(e)
            return Ngrams(inter_list, exclusive = False)
    
    def union(self, *ngrams):
        if self.exclusive:
            union_set = self._ngrams
            for o in ngrams:
                union_set = union_set.union(o._ngrams)
            return Ngrams(union_set, exclusive = True)
        else:
            union_list = deepcopy(self._ngrams)
            for o in ngrams:
                union_list.extend(o._ngrams)
            return Ngrams(union_list, exclusive = False)


def _get_ngrams(n, text, exclusive = True):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = Ngrams(exclusive = exclusive)
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences, exclusive = True):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0
    
    words = _split_into_words(sentences)
    return _get_ngrams(n, words, exclusive = exclusive)


def _len_lcs(x, y):
    """
    Returns the length of the Longest Common Subsequence between sequences x
    and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y, exclusive = True):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = _lcs(x, y)
    
    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)
    
    recon_list = list(map(lambda x: x[0], _recon(i, j)))
    return Ngrams(recon_list, exclusive = exclusive)
    return recon_tuple


def multi_rouge_n(sequences, scores_ids, n = 2, exclusive = True):
    """
    Efficient way to compute highly repetitive scoring
    i.e. sequences are involved multiple time

    Args:
        sequences(list[str]): list of sequences (either hyp or ref)
        scores_ids(list[tuple(int)]): list of pairs (hyp_id, ref_id)
            ie. scores[i] = rouge_n(scores_ids[i][0],
                                    scores_ids[i][1])

    Returns:
        scores: list of length `len(scores_ids)` containing rouge `n`
                scores as a dict with 'f', 'r', 'p'
    Raises:
        KeyError: if there's a value of i in scores_ids that is not in
                  [0, len(sequences)[
    """
    ngrams = [_get_word_ngrams(n, sequence, exclusive = exclusive)
              for sequence in sequences]
    counts = [len(ngram) for ngram in ngrams]
    
    scores = []
    for hyp_id, ref_id in scores_ids:
        evaluated_ngrams = ngrams[hyp_id]
        evaluated_count = counts[hyp_id]
        
        reference_ngrams = ngrams[ref_id]
        reference_count = counts[ref_id]
        
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)
        
        scores += [f_r_p_rouge_n(evaluated_count,
                                 reference_count, overlapping_count)]
    return scores


def rouge_n(evaluated_sentences, reference_sentences,
            n = 2, raw_results = False, exclusive = True):
    """
    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.

    Returns:
      A tuple (f1, precision, recall) for ROUGE-N

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0:
        raise ValueError("Hypothesis is empty.")
    if len(reference_sentences) <= 0:
        raise ValueError("Reference is empty.")
    
    evaluated_ngrams = _get_word_ngrams(
        n, evaluated_sentences, exclusive = exclusive)
    reference_ngrams = _get_word_ngrams(
        n, reference_sentences, exclusive = exclusive)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)
    
    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)
    
    if raw_results:
        o = {
            "hyp": evaluated_count,
            "ref": reference_count,
            "overlap": overlapping_count
        }
        return o
    else:
        return f_r_p_rouge_n(
            evaluated_count, reference_count, overlapping_count)


def f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count):
    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count
    
    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count
    
    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    
    return {"f": f1_score, "p": precision, "r": recall}


def _union_lcs(evaluated_sentences, reference_sentence,
               prev_union = None, exclusive = True):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C.
    For example:
    if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8
    and c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1
    is "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5".
    The union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5"
    and LCS_u(r_i, C) = 4/5.

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      float: LCS_u(r_i, C)

    ValueError:
      Raises exception if a param has len <= 0
    """
    if prev_union is None:
        prev_union = Ngrams(exclusive = exclusive)
    
    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    
    lcs_union = prev_union
    prev_count = len(prev_union)
    reference_words = _split_into_words([reference_sentence])
    
    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs = _recon_lcs(reference_words, evaluated_words, exclusive = exclusive)
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)
    
    new_lcs_count = len(lcs_union) - prev_count
    return new_lcs_count, lcs_union


def rouge_l_summary_level(
        evaluated_sentences, reference_sentences, raw_results = False, exclusive = True, **_):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf

    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = (2*R_lcs*P_lcs) / (R_lcs * P_lcs)

    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary

    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      A float: F_lcs

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    
    # total number of words in reference sentences
    m = len(
        Ngrams(
            _split_into_words(reference_sentences),
            exclusive = exclusive))
    
    # total number of words in evaluated sentences
    n = len(
        Ngrams(
            _split_into_words(evaluated_sentences),
            exclusive = exclusive))
    
    # print("m,n %d %d" % (m, n))
    union_lcs_sum_across_all_references = 0
    union = Ngrams(exclusive = exclusive)
    for ref_s in reference_sentences:
        lcs_count, union = _union_lcs(evaluated_sentences,
                                      ref_s,
                                      prev_union = union,
                                      exclusive = exclusive)
        union_lcs_sum_across_all_references += lcs_count
    
    llcs = union_lcs_sum_across_all_references
    r_lcs = llcs / m
    p_lcs = llcs / n
    
    f_lcs = 2.0 * ((p_lcs * r_lcs) / (p_lcs + r_lcs + 1e-8))
    
    if raw_results:
        o = {
            "hyp": n,
            "ref": m,
            "overlap": llcs
        }
        return o
    else:
        return {"f": f_lcs, "p": p_lcs, "r": r_lcs}
