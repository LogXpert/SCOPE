# SPDX-License-Identifier: MIT
# This file implements the scope algorithm for log parsing.

from abc import ABC, abstractmethod
from typing import cast, Collection, IO, Iterable, List, MutableMapping, MutableSequence, Optional, Sequence, Tuple, \
    TYPE_CHECKING, TypeVar, Union
from enum import Enum
from cachetools import LRUCache, Cache
from itertools import groupby
import random

from simpleProfiler import Profiler, NullProfiler
from collections import defaultdict
import logging
import logging.config
import re
import en_core_web_lg
import math
import json
import requests
import os
from openai import OpenAI
import spacy
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
from modelscope import AutoTokenizer

nlp = en_core_web_lg.load()

prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
infix_re  = compile_infix_regex(nlp.Defaults.infixes)

pattern = r'\b\w+\([^)]+\)\b|\b\w+\[[^\]]+\]\b' # (xxx) and [xxx] as one non-split token

nlp.tokenizer = Tokenizer(
    nlp.vocab,
    rules=nlp.Defaults.tokenizer_exceptions,
    prefix_search=None,
    suffix_search=None,
    infix_finditer=None,
    token_match=re.compile(pattern).match
)

special_words = ["<*>", "<PATH>", "<ID>", "<IP>", "<HEX>", "<DATE>", "<URL>", "<SLOT>", "<NUM>", "<CMD>", "<UNIT>", "-", "<L=3>", "<L=4>"]
for word in special_words:
    nlp.tokenizer.add_special_case(word, [{ORTH: word}])

if os.path.exists("./scope.log"):
    os.rename("./scope.log", "./scope.log.prev")
scope_logger = logging.getLogger("scope_logger")
scope_logger.setLevel(logging.INFO)
scope_handler = logging.FileHandler("./scope.log", mode="w")
scope_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
scope_logger.addHandler(scope_handler)

scope_logger.info("start datetime: %s", str(os.popen("date").read().strip()))

# Define pairs of symmetric symbols
SYMMETRIC_PAIRS = {
    '{': '}',
    '[': ']',
    '(': ')',
    #'<': '>'
}
OPENING = set(SYMMETRIC_PAIRS.keys())
CLOSING = set(SYMMETRIC_PAIRS.values())

class LlmQA:
    """
    Class to store and manage LLM QA results for pairs of log message token lists.
    """

    def __init__(self):
        # Use tuple of (tuple(templateTokens), tuple(newMessageTokens)) as key
        self.qAHistory = {}

    def checkQA(self, templateTokens: Iterable[str], newMessageTokens: Iterable[str]) -> tuple[bool, object]:
        """
        Check if the QA result for the given key exists.
        """
        key = (tuple(templateTokens), tuple(newMessageTokens))
        if key in self.qAHistory:
            return True, self.qAHistory[key]
        return False, None

    def saveQA(self, templateTokens: Iterable[str], newMessageTokens: Iterable[str], result: object) -> None:
        """
        Save the QA information: only keep the latest QA result, replacing any previous one.
        """
        key = (tuple(templateTokens), tuple(newMessageTokens))
        self.qAHistory[key] = result

class Template():
    def __init__(self, templateTokens: Iterable[str] = "This is a default log template", templateId: int = 0, isPosSupported: bool = False, precomputedPosTag: Iterable[str] = None) -> None:
      self.templateId = templateId
      self.matchedLogSize = 1 # there is log matched this template when it's created, so 1 by default
      self.preemptedTokenSet = defaultdict(set) # {0:set(preempted token, <*>), 1:set(), 2:set(), ...} it stores the preempted token which would be assigned to <*> node to preempt existing node token in tree
      self.isPosSupported = isPosSupported
      self.setTemplate(templateTokens)
      self.setTokenPosTag(templateTokens, precomputedPosTag)
      self.isLLMCalculated = False # whether the template is calculated by LLM or not

    def getTemplateStr(self) -> str:
        return ' '.join(self.templateTokens)

    def getTemplateTokens(self) -> str:
        return self.templateTokens

    def getPreemptedTokenSet(self) -> MutableMapping[int, set]:
        return self.preemptedTokenSet

    def setTemplate(self, templateTokens: Iterable[str]) -> None:
        self.templateTokens = templateTokens
        for index, token in enumerate(templateTokens): # start from 0
            self.preemptedTokenSet[index].add(token)

    def increaseMatchedLogSize(self) -> None:
        self.matchedLogSize += 1

    def getMatchedLogSize(self) -> int:
        return self.matchedLogSize

    def getTokenPosTag(self) -> MutableMapping[int, str]:
        return self.tokenPosTag

    def setTokenPosTag(self, templateTokens: Iterable[str], precomputedPosTag: Iterable[str] = None) -> None:
        if precomputedPosTag is not None:
            self.tokenPosTag = precomputedPosTag
        else:
            self.tokenPosTag = self.tokensPosTagger(templateTokens)

    def tokensPosTagger(self, tokens: Iterable[str]) -> Iterable[str]:
        if self.isPosSupported:
            lowercaseTokens = [element.lower() for element in tokens]
            posTagger = nltk.pos_tag(lowercaseTokens)
            tokenPosTag = [pos for token, pos in posTagger]
        else:
            tokenPosTag = ["unknown"] * len(tokens)
        return tokenPosTag

    def setLLMCalculated(self, isLLMCalculated: bool) -> None:
        if isLLMCalculated:
            self.isLLMCalculated = isLLMCalculated # only change from False to True

    def getLLMCalculated(self) -> bool:
        return self.isLLMCalculated

_T = TypeVar("_T")
if TYPE_CHECKING:
    class _LRUCache(LRUCache[int, Optional[Template]]):
        #  see https://github.com/python/mypy/issues/4148 for this hack
        ...
else:
    _LRUCache = LRUCache


class NodeType(Enum):
    ROOT = 1
    DIRECTION = 2
    INTERMEDIATE = 3
    LEAF = 4

class SequenceType(Enum):
    FORWARD = 1
    REVERSE = 2

class Node():
    __slots__ = ["nodeType", "keyToChildNode", "templateIds", "tokensInWildcard"]

    def __init__(self, nodeType: NodeType) -> None:
        self.nodeType: NodeType = nodeType
        self.keyToChildNode: MutableMapping[str, Node] = {}
        self.templateIds: Sequence[int] = set()
        self.tokensInWildcard = set()


class Scope():
    def __init__(self,
                 depth: int = 4,
                 sim_th: float = 0.4,
                 max_children: int = 100,
                 max_clusters: Optional[int] = None,
                 extra_delimiters: Sequence[str] = (),
                 profiler: Profiler = NullProfiler(),
                 param_str: str = "<*>",
                 parametrize_numeric_tokens: bool = True,
                 bi_tree_support: bool = False,
                 pool_support: bool = False,
                 POS_support: bool = False,
                 LLM_support: bool = False,
                 LLM_provider: str = "openai",
                 LLM_model: str = "gpt-3.5-turbo",
                 LLM_api_key: Optional[str] = None,
                 LLM_thinking: bool = False
                 ) -> None:
        if depth < 3:
            raise ValueError("depth argument must be at least 3")

        self.log_cluster_depth = depth
        self.max_node_depth = depth - 2  # max depth of a prefix tree node, starting from zero
        self.sim_th = sim_th
        self.max_children = max_children
        self.root_node = Node(NodeType.ROOT)
        self.profiler = profiler
        self.extra_delimiters = extra_delimiters
        self.max_clusters = max_clusters
        self.param_str = param_str
        self.parametrize_numeric_tokens = parametrize_numeric_tokens
        self.bi_tree_support = bi_tree_support
        self.pool_support = pool_support
        self.POS_support = POS_support
        self.LLM_support = LLM_support
        self.LLM_provider = LLM_provider
        self.LLM_model = LLM_model
        self.LLM_api_key = LLM_api_key
        self.LLM_thinking = LLM_thinking

        self.idToTemplateCluster: MutableMapping[int, Template] = {}
        self.lengthToTemplateIds = defaultdict(list)
        self.templateId = 0
        self.TplUpdByFwdTree = 0
        self.TplUpdByRevTree = 0
        self.TplUpdByPool = 0
        self.totalToken = 0
        self.lengthNodeCount = 0
        self.llmQAHistory = LlmQA()
        self.llmCallCnt = 0
        self.llmCallTokens = 0

    def getNewTemplateId(self) -> int:
        self.templateId += 1
        return self.templateId

    @property
    def clusters(self) -> Collection[Template]:
        return cast(Collection[Template], self.idToTemplateCluster.values())

    @property
    def TplUpdByRevTrees(self) -> int:
        return self.TplUpdByRevTree

    @property
    def TplUpdByFwdTrees(self) -> int:
        return self.TplUpdByFwdTree

    @property
    def TplUpdByPools(self) -> int:
        return self.TplUpdByPool

    @property
    def totalTokens(self) -> int:
        return self.totalToken

    @property
    def lengthNodeCounts(self) -> int:
        return self.lengthNodeCount

    @staticmethod
    def has_numbers(s: Iterable[str]) -> bool:
        return any(char.isdigit() for char in s)

    def print_tree(self, file: Optional[IO[str]] = None, max_clusters: int = 5) -> None:
        self.print_node("root", self.root_node, 0, file, max_clusters)

    def print_node(self, token: str, node: Node, depth: int, file: Optional[IO[str]], max_clusters: int) -> None:
        out_str = '\t' * depth

        if depth == 0:
            out_str += f'<{token}>'
        elif depth == 1:
            if token.isdigit():
                out_str += f'<L={token}>'
            else:
                out_str += f'<{token}>'
        else:
            out_str += f'"{token}"'

        if len(node.templateIds) > 0:
            out_str += f" (cluster_count={len(node.templateIds)})"

        for token, child in node.keyToChildNode.items():
            self.print_node(token, child, depth + 1, file, max_clusters)

        for cid in node.templateIds[:max_clusters]:
            cluster = self.idToTemplateCluster[cid]
            out_str = '\t' * (depth + 1) + str(cluster)

    def split_string(self, s):
        # First, replace any hyphens with spaces
        s = s.replace("-", "__DASH__")
        punctuation_pattern = r'^[,.\!?;:]+|[,.\!?;:]+$'
        match = re.match(punctuation_pattern, s)
        if match:
            front_punctuation = match.group(0)
        else:
            front_punctuation = ''

        stripped_s = re.sub(r'^[,.\!?;:]+', '', s)
        match = re.search(punctuation_pattern, stripped_s)
        if match:
            back_punctuation = match.group(0)
            stripped_s = re.sub(r'[,.\!?;:]+$', '', stripped_s)
        else:
            back_punctuation = ''

        result = []
        if front_punctuation:
            for char in front_punctuation:
                result.append(char)

        if '=' in stripped_s and ':' in stripped_s:
            if stripped_s.index('=') < stripped_s.index(':'):
                left, sep1, remainder = re.split(r"(=)", stripped_s, 1)
                mid, sep2, right = re.split(r"(:)", remainder, 1)
            else:
                left, sep1, remainder = re.split(r"(:)", stripped_s, 1)
                mid, sep2, right = re.split(r"(=)", remainder, 1)
            result.extend([left, sep1, mid, sep2, right])

        elif stripped_s.count('=') == 1 or (stripped_s.count('=') > 1 and not "==" in stripped_s):
            left, sep, right = re.split(r"(=)", stripped_s, 1)
            result.extend([left, sep, right])

        elif stripped_s.count(':') == 1 or (stripped_s.count(':') > 1 and not "::" in stripped_s):
            left, sep, right = re.split(r"(:)", stripped_s, 1)
            result.extend([left, sep, right])

        else:
            result.append(stripped_s)

        if back_punctuation:
            for char in back_punctuation:
                result.append(char)

        tokens = []
        for part in result:
            if (
                len(part) > 1
                and part[0] in SYMMETRIC_PAIRS and part[-1] == SYMMETRIC_PAIRS[part[0]]):
                tokens.append(part[0])
                tokens.append(part[1:-1])
                tokens.append(part[-1])
            elif (len(part) > 1
                and part[0] in SYMMETRIC_PAIRS and not SYMMETRIC_PAIRS[part[0]] in part):
                tokens.append(part[0])
                tokens.append(part[1:])
            elif (len(part) > 1
                and part[-1] in CLOSING and \
                not any(key for key in SYMMETRIC_PAIRS if SYMMETRIC_PAIRS[key] == part[-1] and key in part[0:-1])):
                tokens.append(part[:-1])
                tokens.append(part[-1])
            elif len(part) > 2:
                prefixIdx = suffixIdx = 0
                splitDone = False
                i = 1
                while i < len(part):
                    if part[i] in SYMMETRIC_PAIRS:
                        prefixIdx = i
                        close_sym = SYMMETRIC_PAIRS[part[i]]
                        for k in range(prefixIdx+1, len(part)):
                            if part[k] == close_sym:
                                suffixIdx = k
                                tokens.append(part[0:prefixIdx])
                                tokens.append(part[prefixIdx])
                                tokens.append(part[prefixIdx+1:suffixIdx])
                                tokens.append(part[suffixIdx])
                                tokens.append(part[suffixIdx+1:])
                                splitDone = True
                                break
                        if splitDone == False:
                            tokens.append(part[0:prefixIdx+1])
                            tokens.append(part[prefixIdx+1:])
                            splitDone = True
                    elif part[i] in CLOSING:
                        tokens.append(part[0:i])
                        tokens.append(part[i])
                        tokens.append(part[i+1:])
                        splitDone = True

                    if splitDone == True:
                        break
                    else:
                        i += 1
                if not splitDone:
                    tokens.append(part)
            else:
                tokens.append(part)

        return tokens

    def get_total_cluster_size(self) -> int:
        size = 0
        for c in self.idToTemplateCluster.values():
            size += cast(Template, c).matchedLogSize
        return size

    def get_clusters_ids_for_seq_len(self, seq_fir: Union[int, str]) -> Collection[int]:
        """
        seq_fir: int/str - the first token of the sequence
        Return all clusters with the specified count of tokens
        """

        def append_clusters_recursive(node: Node, id_list_to_fill: MutableSequence[int]) -> None:
            id_list_to_fill.extend(node.templateIds)
            for child_node in node.keyToChildNode.values():
                append_clusters_recursive(child_node, id_list_to_fill)

        cur_node = self.root_node.keyToChildNode.get(str(seq_fir))

        # no template with same token count
        if cur_node is None:
            return []

        target: MutableSequence[int] = []
        append_clusters_recursive(cur_node, target)
        return target

    def add_seq_to_prefix_tree(self, root_node: Node, cluster: Template, seqType: SequenceType) -> None:
        tokens = cluster.getTemplateTokens()
        preemptedTokenSet = cluster.getPreemptedTokenSet()
        token_count = len(tokens)
        token_count_str = str(token_count)
        token_seqType_str = str(int(seqType.value))
        if token_count_str not in root_node.keyToChildNode:
            first_layer_node = Node(NodeType.DIRECTION)
            root_node.keyToChildNode[token_count_str] = first_layer_node
            self.lengthNodeCount += 1
        else:
            first_layer_node = root_node.keyToChildNode[token_count_str]

        if  token_seqType_str not in first_layer_node.keyToChildNode:
            sec_layer_node = Node(NodeType.INTERMEDIATE)
            first_layer_node.keyToChildNode[token_seqType_str] = sec_layer_node
        else:
            sec_layer_node = first_layer_node.keyToChildNode[token_seqType_str]

        cur_node = sec_layer_node

        # handle case of empty log string
        if token_count == 0:
            cur_node.templateIds = [cluster.templateId]
            return

        tokens = tokens[::1] if seqType == SequenceType.FORWARD else tokens[::-1]

        #tokens = tokens.split()
        def get_half_tokens(tokens):
            if token_count % 2 == 0:
                return tokens[:token_count // 2 + 1]
            else:
                return tokens[:(token_count + 1) // 2]
        tokens = get_half_tokens(tokens)

        current_depth = 1
        for index, token in enumerate(tokens):
            index = index if seqType == SequenceType.FORWARD else token_count - index - 1
            if token == self.param_str:
                if self.param_str not in cur_node.keyToChildNode: # * node always can be added as room is reserved
                    new_node = Node(NodeType.INTERMEDIATE)
                    new_node.tokensInWildcard = preemptedTokenSet[index] # <*> node is new created, set node.tokensInWildcard = preempted tokens of template
                    cur_node.keyToChildNode[self.param_str] = new_node
                    cur_node = new_node
                else:
                    new_node = cur_node.keyToChildNode[self.param_str]
                    new_node.tokensInWildcard = new_node.tokensInWildcard | preemptedTokenSet[index] # <*> node exists, add preempted tokens of template into node.tokensInWildcard
                    cur_node = new_node
            else:
                if token not in cur_node.keyToChildNode:
                    if self.parametrize_numeric_tokens and self.has_numbers(token): # it's a parameter token
                        if self.param_str not in cur_node.keyToChildNode:
                            new_node = Node(NodeType.INTERMEDIATE)
                            #new_node.tokensInWildcard.append(token) # <*> node is used to preempt existing token in tree
                            cur_node.keyToChildNode[self.param_str] = new_node
                            cur_node = new_node
                        else:
                            cur_node = cur_node.keyToChildNode[self.param_str] # <*> is selected if token is not in tree
                    else: # it's a normal token
                        if self.param_str in cur_node.keyToChildNode: # <*> node exists in tree, add token to it if room is available
                            if len(cur_node.keyToChildNode) < self.max_children:
                                new_node = Node(NodeType.INTERMEDIATE)
                                cur_node.keyToChildNode[token] = new_node
                                cur_node = new_node
                            else:
                                cur_node = cur_node.keyToChildNode[self.param_str] # if the number of children is full, use * node directly
                        else: # <*> node not exists in tree
                            if len(cur_node.keyToChildNode) + 1 < self.max_children: # there is room for new token if at least one room is reserved for <*>
                                new_node = Node(NodeType.INTERMEDIATE)
                                cur_node.keyToChildNode[token] = new_node
                                cur_node = new_node
                            else: # only 1 room left, only can add <*> node
                                new_node = Node(NodeType.INTERMEDIATE)
                                #new_node.tokensInWildcard.append(token) #<*> node is used to preempt existing token in tree
                                cur_node.keyToChildNode[self.param_str] = new_node
                                cur_node = new_node
                else: # if the token is matched
                    cur_node = cur_node.keyToChildNode[token]
            current_depth += 1

            if current_depth >= self.max_node_depth or current_depth > len(tokens):
                # clean up stale clusters before adding a new one.
                new_cluster_ids = set()
                for cluster_id in cur_node.templateIds:
                    if cluster_id in self.idToTemplateCluster:
                        new_cluster_ids.add(cluster_id)
                new_cluster_ids.add(cluster.templateId)
                cur_node.templateIds = new_cluster_ids
                cur_node.nodeType = NodeType.LEAF
                break

    # seq1 is a template, seq2 is the log to match
    def get_seq_distance(self, seq1: Sequence[str], seq2: Sequence[str], \
                         preemptedTokens, templatePosTag, include_params: bool, sim_th: int, newLogPosTag: Optional[Sequence[str]] = None) -> Tuple[float, int]:
        assert len(seq1) == len(seq2)

        # sequences are empty - full match
        if len(seq1) == 0:
            return 1.0, 0

        sim_tokens = 0
        param_count = 0

        # Static POS tags for spaCy
        spacy_static_pos_tags = [
            #"NOUN",  
            "VERB",  
            #"ADJ",    
            #"ADV",   
            "PRON",  
            "DET",    
            "ADP",   
            #"NUM",   
            "CCONJ",  
            "SCONJ",  
            "PART",   
            "AUX",    
            "INTJ",   
            #"PROPN",  
            "PUNCT", 
            "SYM",   
            #"X"       
        ]
        equivalent_pos_groups = [
            {"ADJ","ADV","NOUN","PROPN"},  
            {"CCONJ", "SCONJ"},
        ]

        def is_same_pos_class(pos1, pos2):
            return any(pos1 in group and pos2 in group for group in equivalent_pos_groups)

        def is_determinated_static_part(posTag: str) -> bool:
            if posTag in spacy_static_pos_tags:
                return True
            else:
                return False

        def isAllAlphaCapital(token: str) -> bool:
            if token == "<*>":
                return False

            if any(ch.isdigit() for ch in token):
                return False

            return all(char.isupper() for char in token if char.isalpha())

        def isSnakeMode(token: str) -> bool:
            """
            Check if the token is in snake_case format.
            """
            return bool(re.match(r'^[a-z]+(_[a-z]+)+$', token))

        def isCamelMode(token: str) -> bool:
            """
            Check if the token is in camelCase format.
            """
            return bool(re.match(r'^[a-zA-Z]+([A-Z][a-z0-9]+)+(\(\))?$', token))

        def is_snake_or_camel(identifier):
            snake_case_pattern = r'^[a-z]+(_[a-z]+)+$'
            camel_case_pattern = r'^[a-zA-Z]+([A-Z][a-z0-9]+)+$'
            return bool(re.match(snake_case_pattern, identifier) or re.match(camel_case_pattern, identifier))

        for index, (tplToken, newToken) in enumerate(zip(seq1, seq2)):
            if tplToken == newToken:
                if not re.match(r"<[^:]+>", tplToken):
                    sim_tokens += 1
                else:
                    param_count += 1
            elif re.match(r"<[^:]+>", tplToken):# or re.match(r"<[^:]+>", newToken):
                if newToken in preemptedTokens[index]:
                    sim_tokens += 1
                else:
                    param_count += 1
            elif re.match(r'(([A-Za-z_][A-Za-z0-9_]*)(::[A-Za-z_][A-Za-z0-9_]*)+)', newToken) \
                  or re.match(r'(([A-Za-z_][A-Za-z0-9_]*)(::[A-Za-z_][A-Za-z0-9_]*)+)', tplToken):
                return 0.0, 0
            elif index>0 and index<len(seq1)-1 and (not self.has_numbers(newToken) and seq2[index+1] in {"=", ":"}): # and (seq2[index-1] in {",", ".", ":"}):
                return 0.0, 0
            elif ((not self.has_numbers(newToken) and not self.has_numbers(tplToken)) \
                  and ((index == 0 and (isSnakeMode(newToken) or isSnakeMode(tplToken) or isCamelMode(tplToken))) \
                       or (index>0 and not seq1[index-1] in {"=", ":"} and (isCamelMode(newToken) or isCamelMode(tplToken))))):
                return 0.0, 0
            elif isAllAlphaCapital(tplToken) and isAllAlphaCapital(newToken) and (index == 0 or not seq1[index-1] in {"=", ":"}):
                return 0.0, 0
            elif self.POS_support:
                # If newLogPosTag was not provided, generate it from seq2
                if newLogPosTag is None:
                    lowercaseTokens = [element.lower() for element in seq2]
                    posTagger = nltk.pos_tag(lowercaseTokens)
                    destTokenPosTag = [pos for token, pos in posTagger]
                else:
                    destTokenPosTag = newLogPosTag
                if not '<*>' in {tplToken, newToken} \
                    and is_determinated_static_part(templatePosTag[index]) == True:
                    if is_determinated_static_part(templatePosTag[index]) != is_determinated_static_part(destTokenPosTag[index]):
                        return 0.0, 0
                    elif tplToken != newToken:
                        return 0.0, 0

                if not '<*>' in {tplToken, newToken} \
                    and templatePosTag[index][0:1] != destTokenPosTag[index][0:1] \
                    and not is_same_pos_class(templatePosTag[index], destTokenPosTag[index]) \
                    and (index == 0 or not seq1[index-1] in {"=", ":"}):
                    return 0.0, 0

        maxParamCount = round(len(seq1) * sim_th)+1
        if param_count > maxParamCount: #avoid too many <*> are matched
            return 0.0, 0

        if include_params:
            sim_tokens += param_count

        ret_val = float(sim_tokens) / len(seq1)

        return ret_val, param_count

    def create_template(self, seq1: Sequence[str], seq2: Sequence[str]) -> Sequence[str]:
        """
        Loop through two sequences and create a template sequence that
        replaces unmatched tokens with the parameter string.

        :param seq1: first sequence
        :param seq2: second sequence
        :return: template sequence with param_str in place of unmatched tokens
        """
        assert len(seq1) == len(seq2)
        return [token2 if token1 == token2 else self.param_str for token1, token2 in zip(seq1, seq2)]

    def fast_match(self,
                   templateIds: Collection[int],
                   tokens: Sequence[str],
                   sim_th: float,
                   include_params: bool,
                   tokenPosTag: Sequence[str]) -> tuple[Optional[Template], float, float, Sequence[str]]:
        """
        Find the best match for a log message (represented as tokens) versus a list of clusters
        :param templateIds: List of clusters to match against (represented by their IDs)
        :param tokens: the log message, separated to tokens.
        :param sim_th: minimum required similarity threshold (None will be returned in no clusters reached it)
        :param include_params: consider tokens matched to wildcard parameters in similarity threshold.
        :return: Best match cluster or None
        """
        match_cluster = None

        max_sim: Union[int, float] = -1
        max_param_count = -1
        max_cluster = None
        matched_templateId = 0
        llm_template = None


        if sim_th == 0.0:
            sim_th = 1 - (math.log(len(tokens), 2) / len(tokens))
            #sim_th = int(sim_th * 10) / 10
            sim_th = round(sim_th, 1)

        for id in templateIds:
            # Try to retrieve cluster from cache with bypassing eviction
            # algorithm as we are only testing candidates for a match.
            cluster = self.idToTemplateCluster.get(id)
            if cluster is None:
                continue
            cur_sim, param_count = self.get_seq_distance(cluster.getTemplateTokens(), tokens, cluster.getPreemptedTokenSet(), cluster.getTokenPosTag(), include_params, sim_th, tokenPosTag)
            if cur_sim > max_sim or (cur_sim == max_sim and param_count < max_param_count): # prefer log with minimam para count when sim value is same
                max_sim = cur_sim
                max_param_count = param_count
                max_cluster = cluster
                matched_templateId = id
        if max_sim >= sim_th:
            if not self.LLM_support:
                match_cluster = max_cluster
            else:
                if max_sim != 1: #and max_cluster.getLLMCalculated() == False: #if LLm has calculated, max_sim should be 1?
                    template = max_cluster.getTemplateTokens()
                    newMsg = self.wildcardApplyInMsgBasedOnTemplate(template, tokens)
                    (isHandled, result) = self.llmQAHistory.checkQA(template, newMsg)
                    if isHandled:
                        if result is not None: # "" means different template
                            llm_template = result
                            match_cluster = max_cluster
                    else:
                        (isSameTemplate, newTemplate) = self.isSameTemplateByLLM(template, newMsg)
                        if isSameTemplate:
                            if len(template) == len(newTemplate):
                                match_cluster = max_cluster
                                llm_template = newTemplate
                            else:
                                scope_logger.error(f"LLM returns different length template, reuse original one. len(original template): {len(template)}, len(new template): {len(newTemplate)}")
                                llm_template = template
                                match_cluster = max_cluster
                        self.llmQAHistory.saveQA(template, newMsg, llm_template) #store the LLM result for future use
                else:
                    match_cluster = max_cluster

        return match_cluster, max_sim, max_param_count, llm_template

    def wildcardApplyInMsgBasedOnTemplate(self, templateList: Sequence[str], msgList: Sequence[str]) -> Sequence[str]:
        return [self.param_str if self.param_str in templateList[i] else token for i, token in enumerate(msgList)]

    def findMatchedTemplateFromPool(self, length, tokens: Sequence[str], tokenPosTag: Sequence[str]) -> Tuple[Optional[Template], Optional[Sequence[str]]]:
        if length not in self.lengthToTemplateIds:
            return None, None
        templateIds = self.lengthToTemplateIds.get(length)
        matchedTemplate, _, _, llmTemplate = self.fast_match(templateIds, tokens, self.sim_th, True, tokenPosTag)
        return matchedTemplate, llmTemplate

    def updateTemplateOfPool(self, template: Template, newTemplateTokens: Sequence[str], isLLMCalc: bool) -> None:
        template.setTemplate(newTemplateTokens)
        template.increaseMatchedLogSize()
        if isLLMCalc:
            template.setLLMCalculated(True)
        # lst = self.lengthToTemplateIds[len(newTemplateTokens)]
        # if template.templateId in lst:
        #     lst.append(lst.pop(lst.index(template.templateId)))

    def findMatchedTemplateFromTree(self,
                    root_node: Node,
                    tokens: Sequence[str],
                    sim_th: float,
                    include_params: bool, tokenPosTag: Sequence[str]) -> tuple[
                        tuple[Optional[Template], float, float, Optional[Sequence[str]]],
                        tuple[Optional[Template], float, float, Optional[Sequence[str]]]
                    ]:
        result = self.tree_search(root_node, tokens, SequenceType.FORWARD, sim_th, include_params, tokenPosTag)
        if result is not None:
            fw, fw_sim, fw_paraCnt, fw_llmTemplate = result
        else:
            fw, fw_sim, fw_paraCnt, fw_llmTemplate = None, 0.0, 0.0, None
        if self.bi_tree_support:
            result =  self.tree_search(root_node, tokens, SequenceType.REVERSE, sim_th, include_params, tokenPosTag)
            if result is not None:
                rv, rv_sim, rv_paraCnt, rv_llmTemplate = result
            else:
                rv, rv_sim, rv_paraCnt, rv_llmTemplate = None, 0.0, 0.0, None
        else: # search reverse tree but not use reverse data for better comparision with bi-tree mode
            #result = self.tree_search(root_node, tokens, SequenceType.REVERSE, sim_th, include_params, tokenPosTag)
            #rv, rv_sim, rv_paraCnt, rv_llmTemplate = fw, fw_sim, fw_paraCnt, fw_llmTemplate
            rv, rv_sim, rv_paraCnt, rv_llmTemplate = None, 0.0, 0.0, None

        return (fw, fw_sim, fw_paraCnt, fw_llmTemplate), (rv, rv_sim, rv_paraCnt, rv_llmTemplate)

    def tree_search(self,
                    root_node: Node,
                    tokens: Sequence[str],
                    seq_type: SequenceType,
                    sim_th: float,
                    include_params: bool,
                    tokenPosTag: Sequence[str]) -> tuple[Optional[Template], float, float, Optional[Sequence[str]]]:

        # at first level, children are grouped by token (word) count
        token_count = len(tokens)
        cur_node = root_node.keyToChildNode.get(str(token_count))
        # no template with same token count yet
        if cur_node is None:
            return None, 0.0, 0.0, None

        cur_node = cur_node.keyToChildNode.get(str(int(seq_type.value)))
        # no template with matched dirction sequence yet
        if cur_node is None:
            return None, 0.0, 0.0, None
        # handle case of empty log string - return the single cluster in that group
        if token_count == 0:
            return self.idToTemplateCluster.get(cur_node.templateIds[0]), 0.0, 0.0, None

        # find the leaf node for this log - a path of nodes matching the first N tokens (N=tree depth)
        tokensToMatch = tokens[::1] if seq_type == SequenceType.FORWARD else tokens[::-1]

        for token in tokensToMatch:
            keyToChildNode = cur_node.keyToChildNode
            token_node = keyToChildNode.get(token)
            wildcard_node = keyToChildNode.get(self.param_str)
            if token_node is not None and wildcard_node is not None: # checke whether <*> has preempted token
                if token in wildcard_node.tokensInWildcard:
                    cur_node = wildcard_node
                else:
                    cur_node = token_node
            elif token_node is not None:
                cur_node = token_node
            elif wildcard_node is not None:
                cur_node = wildcard_node
            else:
                return None, 0.0, 0.0, None
            # token is matched:
            if cur_node.nodeType == NodeType.LEAF:
                break

        # get best match among all clusters with same prefix, or None if no match is above sim_th
        cluster, sim, paraCnt, llmTemplate = self.fast_match(cur_node.templateIds, tokens, sim_th, include_params, tokenPosTag)
        return cluster, sim, paraCnt, llmTemplate

    def buildTemplateWithInputLog(self, length, tokens: Sequence[str], tokenPosTag: Sequence[str] = None) -> Optional[Template]:
        template = Template(tokens, self.getNewTemplateId(), isPosSupported=self.POS_support, precomputedPosTag=tokenPosTag)
        self.idToTemplateCluster[template.templateId] = template
        #self.lengthToTemplateIds[length].append(template.templateId)
        self.lengthToTemplateIds[length].insert(0, template.templateId)
        return template

    def addTemplateSeqToPrefixTree(self, root_node: Node, template: Template) -> None:
        self.add_seq_to_prefix_tree(root_node, template, SequenceType.FORWARD)
        if self.bi_tree_support:
            self.add_seq_to_prefix_tree(root_node, template, SequenceType.REVERSE)
        else: # add reverse sequence to tree but not used for better comparision with bi-tree mode
            self.add_seq_to_prefix_tree(root_node, template, SequenceType.REVERSE)


    def add_log_message(self, content: str) -> Tuple[Template, str]:
        #content_tokens = self.get_content_as_tokens(content)
        content_tokens, tokenPosTag = self.extractTokensOfMsg(content)

        # matchedPoolTemplate = self.buildTemplateWithInputLog(len(content_tokens), content_tokens)
        # update_type = "created"
        # return matchedPoolTemplate, update_type
        length = len(content_tokens)
        if self.profiler:
            self.profiler.start_section("findMatchedTemplateFromTree")
        (fwSeqMatchedTemplate, fwSim, fwParaCnt, fwllmTemplate), (RvSeqMatchedTemplate, rvSim, rvParaCnt, rvllmTemplate) \
                                    = self.findMatchedTemplateFromTree(self.root_node, content_tokens, self.sim_th, include_params=True, tokenPosTag=tokenPosTag)
        if self.profiler:
            self.profiler.end_section()

        # Match no existing template
        if fwSeqMatchedTemplate is None and RvSeqMatchedTemplate is None: # both forward and reverse sequence don't have matched template
            if self.pool_support:
                if self.profiler:
                    self.profiler.start_section("findMatchedTemplateFromPool")
                matchedPoolTemplate, llmTemplate = self.findMatchedTemplateFromPool(length, content_tokens, tokenPosTag)
                #matchedPoolTemplate = None
                if self.profiler:
                    self.profiler.end_section()
            else:
                matchedPoolTemplate = None
                llmTemplate = None

            if matchedPoolTemplate is None: # it's a new message which don't have template in pool yet
                matchedPoolTemplate = self.buildTemplateWithInputLog(length, content_tokens, tokenPosTag)
                update_type = "created"
            else: # similar template is found in pool but not in tree, need update template and add to tree
                #newTemplateTokens = self.create_template(content_tokens, matchedPoolTemplate.getTemplateTokens())
                self.TplUpdByPool += 1
                newTemplateTokens = llmTemplate if llmTemplate is not None else self.create_template(content_tokens, matchedPoolTemplate.getTemplateTokens())
                self.updateTemplateOfPool(matchedPoolTemplate, newTemplateTokens, True) # llmTemplate is used if True
                update_type = "updated"

            if self.profiler:
                self.profiler.start_section("addTemplateSeqToPrefixTree")
            self.addTemplateSeqToPrefixTree(self.root_node, matchedPoolTemplate)
            if self.profiler:
                self.profiler.end_section()

        else: # Match existing template at least one direction of tree
            matchedTree = -1 # -1 means no matched tree, 0 means forward tree matched, 1 means reverse tree matched
            if fwSeqMatchedTemplate is not None and RvSeqMatchedTemplate is not None:
                if(fwSeqMatchedTemplate.templateId != RvSeqMatchedTemplate.templateId):
                    #matchedPoolTemplate = fwSeqMatchedTemplate if fwSim > rvSim else RvSeqMatchedTemplate
                    if fwSim > rvSim:
                        matchedPoolTemplate = fwSeqMatchedTemplate
                        llmTemplate = fwllmTemplate
                    elif fwSim < rvSim:
                        matchedPoolTemplate = RvSeqMatchedTemplate
                        llmTemplate = rvllmTemplate
                    elif fwSim == rvSim:
                        if fwParaCnt <= rvParaCnt:
                            matchedPoolTemplate = fwSeqMatchedTemplate
                            llmTemplate = fwllmTemplate
                        else:
                            matchedPoolTemplate = RvSeqMatchedTemplate
                            llmTemplate = rvllmTemplate
                    assert(1)
                else:
                    matchedPoolTemplate = fwSeqMatchedTemplate
                    llmTemplate = fwllmTemplate
                #assert(fwSeqMatchedTemplate.templateId == RvSeqMatchedTemplate.templateId)
            elif fwSeqMatchedTemplate is not None:
                matchedPoolTemplate = fwSeqMatchedTemplate
                llmTemplate = fwllmTemplate
                matchedTree = 0
            else:
                matchedPoolTemplate = RvSeqMatchedTemplate
                llmTemplate = rvllmTemplate
                matchedTree = 1

            isLLMCalc = False
            if llmTemplate is not None:
                newTemplateTokens = llmTemplate
                isLLMCalc = True
            else:
                newTemplateTokens = self.create_template(content_tokens, matchedPoolTemplate.getTemplateTokens())

            if newTemplateTokens != matchedPoolTemplate.getTemplateTokens():
                update_type = "updated"
                if self.profiler:
                    self.profiler.start_section("updateTemplateOfPoolAndTree")

                self.updateTemplateOfPool(matchedPoolTemplate, newTemplateTokens, isLLMCalc)
                self.addTemplateSeqToPrefixTree(self.root_node, matchedPoolTemplate)
                if matchedTree == 1:
                    self.TplUpdByRevTree += 1
                if matchedTree == 0:
                    self.TplUpdByFwdTree += 1

                if self.profiler:
                    self.profiler.end_section()
            else:
                update_type = "none"
                matchedPoolTemplate.increaseMatchedLogSize()
                matchedPoolTemplate.setLLMCalculated(isLLMCalc)

        return matchedPoolTemplate, update_type

    def isSameTemplateByLLM(self, template_tokens: Sequence[str], log_tokens: Sequence[str]) -> Tuple[bool, Sequence[str]]:
        """Check if two tokenized logs belong to the same template using LLM."""
        log_a_tokens = [str(random.randint(1, 100)) if token == '<*>' else token for token in template_tokens]
        log_b_tokens = [str(random.randint(1, 100)) if token == '<*>' else token for token in log_tokens]
        # Construct prompt
        prompt = f"""
        You are a precise log template abstraction engine. You will be given two log messages, each already split into a list of tokens. Your goal is to determine whether two log messages share the **same log template**. Follow the steps strictly and output only the required format.

        ## Input Format:
        Two token lists, each representing a log message.

        ## Step 1: Template Abstraction
        Independently abstract a template from `LogA` and `LogB` using your internal knowledge and following rules, **no comparison** between LogA and LogB when do template abstration seperately.

        ### Template Abstraction Rules

        #### General Rules
        - Some messages may **contain no variables**; do not forcibly abstract in those cases.
        - The **existence of "<*>"** in a message **should not dictate whether everything is abstracted**. Each token should be evaluated individually.
        - **Do not partially abstract inside a token**.
        - The number of tokens in the template must be the same as the input log. **Do not add new tokens or merge multiple tokens into one**.
        Examples:
            ```
            Incorrect: `brightOut` → `bright<*>`, `mapred.job.id` → `mapred.job.<*>`.
            Correct: `brightOut` → `brightOut`, `mapred.job.id` → `mapred.job.id`.
            ```
        ---

        #### Non-Variable (Should Be Constant)
        - If a token is an **adjective, adverb, verb, or proper noun or domain-specific term(e.g., `HTTPS`, `IPv4`, `BSSID`, `SCREEN_ON`, `SCREEN_OFF`, `JOB_SETUP`)**, keep it as a constant.
        - If a token is a **modifier in a compound noun**(e.g., `Removable`, `illegal`, `Idle`, `Active`) and the modifier **represents a fixed attribute or label**, keep the compound noun as a constant.
        - If a token serves as the subject in a sentence with a subject-verb-object structure(e.g., `syslogd`, `Failed none`), then it must be treated as **constants** and not abstracted.
        - If a token **reflects behavioral information(e.g., `Accepted`, `Failed`, `Timeout`)**  — keep it as a constant, especially when there are few semantic variants (typically < 3).
        - If a token **reflects different service outcomes with clear semantic distinctions and belongs to a small set of valid values (typically fewer than 3)**, it must be treated as a constant — **except when it appears as the value in a key:value or key=value pattern, in which case abstraction rules for value tokens apply**.
        - If a token **reflects distinctly different or opposing behaviors** (e.g., `boot` vs `shutdown`) — keep it as a constant.        ---
        - Do not treat "user" and "users", "service" and "services", or other singular/plural variants as the same — keep it as a constant.
        Examples:
            ```
            Failed none  → Failed none
            ```
        ---

        #### Variable (Should Be Abstracted)
        - Replace variable parts (IDs, numbers, IPs, timestamps, user names, etc.) with wildcards `<*>`, e.g., `a1`, `B2` etc.
        - If a token sequence matches the pattern <keyword> <value>, where the keyword is a known category(like `domain`, `interface`, `user`, etc.), then keep the keyword as a constant and abstract the value as `<*>` no matter whehter the value is domain-specific term.
        - If a token is **an instance of a known category with a fixed key and a variable value**, keep the key and abstract its value no matter whehter the value is domain-specific term.
        Examples:
            ```
            domain cluster_root_backup  → domain <*>
            interface eth0              → interface <*>
            user root                   → user <*>
            ```
        - If **a sequence of consecutive key-value pairs includes at least a single abstracted value**, then **all the values in that sequence should be abstracted while retaining their keys**.
        - If a token appears as the value in a key:value or key=value pair, abstract it as <*> while keeping the key constant — unless the value is domain-specific (e.g., SCREEN_ON) or a proper noun.
        - If the value is a boolean (e.g., true, false), always abstract it as <*>, keeping the key unchanged.
        Examples:
            ```
            user=root       → user=<*>
            group=admin     → group=<*>
            ret:false       → ret:<*>
            isOverlap:true  → isOverlap:<*>
            ```
        ---

        #### Summary:
        - **Always retain semantic information and structure.**
        - **Only abstract components that are truly variable.**
        - **Never add or delete tokens.**
        - **Process each message individually.**
        - **Do not partially abstract inside a token, Do not add new tokens or merge multiple tokens into one**.

        ### Template Abstraction for first log message.
        Abstract the template for `LogA` using your internal knowledge and **Template Abstraction Rules** strictly, generate TemplateA and Explanation A. **Don't** consider second log message `LogB` in this step.
            give out explanation with following format:
            Output-1 Format:
            {{
                "LogA": {log_a_tokens},
                "Explanation A": <short explanation of the template abstraction for `LogA`>,
                "TemplateA": [your abstracted template]
            }}

        ### Template Abstraction for second log message.
        Abstract the template for `LogB` using your internal knowledge and **Template Abstraction Rules** strictly, generate TemplateB and Explanation B. **Don't** consider first log message `LogA` in this step.
            give out explanation with following format:
            Output-2 Format:
            {{
                "LogB": {log_b_tokens},
                "TemplateB": [your abstracted template]
                "Explanation B": <short explanation of the template abstraction for `LogB`>
            }}

        ## Step 2: Token-by-Token Template Comparison
        For each position `i` in both templates(TemplateA and TemplateB):
        - If `TemplateA[i] == TemplateB[i]`, treat as match.
        - If both are wildcards (`<*>`), treat as match.
        - If both tokens are constants and they are different, treat as mismatch.
        - If one is constant and one is wildcard → treat as match.
            give out explanation with following format:
            Output-3 Format:
            {{
                "position":
                "tokens":
                "Explanation":[how the decision is made, why the rule is applied, etc.]
                "result":
            }}
        ## Step 2.5: Template Self-Validation
        Before finalizing output, validate each template to ensure the template abstraction alignment with the rules.

        ## Step 3: Final Decision
        - If no violations occurred in Step 2, output `Decision: YES`.
        - If any mismatch of constants is found, output `Decision: NO`.
        - If the logs share the same template, output the abstracted template.
        - If the logs do not share the same template, output an empty template.
            Give out final result with following format:
            Output-4 Format:
            If the two logs share the same template(Decision is YES), return JSON format:
            {{
            "template": [your abstracted template from Step 3],
            }}
            If the two logs do **NOT** share the same template(Decision is NO), return JSON format:
            {{
            "template": []
            }}

        ## Input Logs:
        LogA = {log_a_tokens}
        LogB = {log_b_tokens}
        ---

        Please generate ordered Output-1, Output-2, Output-3, Output-4 result as described above with strictly valid JSON (not Python-style), parseable by json.loads:
        Output-1: <your JSON result for Step 1>
        Output-2: <your JSON result for Step 2>
        Output-3: <your JSON result for Step 3>
        Output-4: <your JSON result for Step 4>

        """

        response = self.call_llm(prompt)
        scope_logger.info(f"LLM JSON response: {response}")
        print("==== LLM JSON response ===")
        print(f"LLM JSON response: {response}")
        print("===========================")

        try:
            # If response is a string containing JSON
            if isinstance(response, str):
                try:
                    # Try to parse it as JSON
                    parsed_response = json.loads(response)
                    template = parsed_response.get("template", "")
                except json.JSONDecodeError:
                    # If it's not valid JSON, try to extract JSON using regex
                    matches = re.findall(r'\{.*?\}', response, re.DOTALL)
                    if matches:
                            json_str = matches[-1]
                            parsed_response = json.loads(json_str)
                            template = parsed_response.get("template", "")
                    else:
                        scope_logger.error(f"Could not find JSON in response: {response}")
                        return False, ""
            # If response is already a dictionary
            elif isinstance(response, dict):
                template = response.get("template", "")
            else:
                scope_logger.error(f"Unexpected response type: {type(response)}")
                return False, ""

            if template:
                # Process template - replace any token containing '<*>' with just '<*>'
                processed_template = []
                for token in template:
                    if '<*>' in token:
                        # part token is abstracted to <*> which leads to POS issue in later comparsion
                        processed_template.append('<*>')
                    else:
                        processed_template.append(token)
                template = processed_template
                return True, template  # Split template string into tokens
            else:
                return False, ""
        except Exception as e:
            scope_logger.error(f"Failed to process LLM response: {e}")
            scope_logger.error(f"Response was: {response}")
            return False, ""

    def extract_last_json(self, text):
        stack = []
        start_index = None
        last_json = None

        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start_index = i
                stack.append(ch)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start_index is not None:
                        last_json = text[start_index:i+1]
        return last_json

    import json

    def parse_llm_stream(self, response):
        import json

        reasoning_log = []
        answer_log = []

        buffer = ""

        for chunk in response:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line.startswith("data:"):
                    continue

                json_str = line[len("data:"):].strip()
                if json_str == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(json_str)
                except Exception as e:
                    print("JSON decode error:", e)
                    continue

                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                reasoning_chunk = delta.get("reasoning_content", "")
                answer_chunk = delta.get("content", "")

                if reasoning_chunk:
                    reasoning_log.append(reasoning_chunk)
                if answer_chunk:
                    answer_log.append(answer_chunk)

                if chunk_data.get("choices", [{}])[0].get("finish_reason") == "stop":
                    break

        return {
            "answer": ''.join(answer_log)
        }

    def call_llm(self, prompt: str) -> str:
        """Call an LLM API with the given prompt and return the response."""
        try:
            client = OpenAI(
                api_key=self.LLM_api_key,
                base_url=self.LLM_provider
            )

            if self.LLM_thinking:
                # Enable thinking mode if configured
                extra_body = {
                    "enable_thinking": True,
                    "thinking_budget": 4096,  # Set budget for thinking tokens
                    "stream": True  # Enable streaming for simplicity
                }
            else:
                # Disable thinking mode if not configured
                extra_body = {
                    "enable_thinking": False,
                    "stream": False  # Disable streaming for simplicity
                }
            messages = [
                {"role": "system", "content": "You are a log parse assistant to help extract log template from two logs."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model = self.LLM_model,
                messages=messages,
                temperature = 0,
                extra_body=extra_body
            )
            if not response:
                scope_logger.error("No response received from LLM API")
                return '{"template": ""}'

            log_content = ""
            if self.LLM_thinking:
                result = self.parse_llm_stream(response)
                log_content = result["answer"]
            else:
                # If not in thinking mode, just get the response content directly
                log_content = response.choices[0].message.content
            #print("LLM response contents: %s" % log_content)
            scope_logger.info("LLM response contents: %s", log_content)
            self.llmCallCnt += 1

            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            prompt_token_ids = encoder.encode(prompt)
            scope_logger.info("Prompt token IDs: %s", len(prompt_token_ids))
            response_token_ids = encoder.encode(log_content)
            scope_logger.info("Response token IDs: %s", len(response_token_ids))
            self.llmCallTokens += (len(response_token_ids) + len(prompt_token_ids))
            scope_logger.info("LLM call tokens: %s", self.llmCallTokens)
            # Try to parse JSON
            try:
                import re
                import json
                last_json_str = self.extract_last_json(log_content)
                scope_logger.info("last_json_str: %s", last_json_str)
                if last_json_str:
                    parsed_response = json.loads(last_json_str)
                    template = parsed_response.get("template", "")
                    parsed = {"template": template}
                else:
                    scope_logger.error(f"Could not find JSON in response: {log_content}")
                    return False, ""
            except (json.JSONDecodeError, TypeError) as e:
                print(f"\nWarning: Could not parse response as JSON: {e}")
                parsed = {"template": ""}
            return parsed

        except Exception as e:
            scope_logger.error(f"Error calling LLM: {str(e)}")
            return '{"template": ""}'

    def extractTokensOfMsg(self, msg: str) -> Tuple[List[str], List[str]]:
        """
        Analyze text from input file using spaCy and NLTK.
        First preprocess special tokens, then analyze with spaCy.

        Args:
            msg (str): The message to analyze

        Returns:
            Tuple[List[str], List[str]]: Final tokens and their POS tags
        """
        msg = msg.replace(",", ", ")
        initial_tokens = msg.split()

        # Process each token to handle special tokens and key-value patterns
        processed_tokens = []
        if len(initial_tokens) > 1:  # Only process if we have more than one token
            for token in initial_tokens:
                # Check if token contains any special token
                contains_special = False
                for special in ['<*>']:  # special_words:
                    if special in token:  # <*> is included into token
                        # token == <*>
                        if token == special:
                            processed_tokens.append(token)
                            contains_special = True
                            break

                        # token cover <*>, split token with pattern like 'key=<HEX>' or 'key:<HEX>'
                        elif '=' in token or ':' in token or '{' in token or '}' in token or '(' in token or ')' in token \
                            or '[' in token or ']' in token or ',' in token or '.' in token:
                            # Split on '=' or ':' first
                            parts = self.split_string(token)
                            # Replace any part that contains a special token
                            for i, part in enumerate(parts):
                            #for sp in special_words:
                                for sp in ['<*>']:
                                    if sp in part and part != sp:  # part has <*> inside, change part to <*>
                                        parts[i] = sp
                            processed_tokens.extend(parts)
                            contains_special = True
                            break
                        elif re.search(r'\.{2,}', token):
                            # Split the token by ellipsis pattern, but keep the ellipsis parts
                            parts = re.split(r'(\.{2,})', token)
                            # Filter out empty strings from the result
                            parts = [part for part in parts if part]
                            for i, part in enumerate(parts):
                                #for sp in special_words:
                                for sp in ['<*>']:
                                    if sp in part and part != sp:
                                        parts[i] = sp
                            processed_tokens.extend(parts)
                            contains_special = True
                            break
                        # If token contains special token but isn't exactly it
                        else:
                            processed_tokens.append(special)
                            #processed_tokens.append(token)
                            contains_special = True
                            break

                # If token doesn't contain any special token, process it normally
                if not contains_special:
                    parts = self.split_string(token)
                    processed_tokens.extend(parts)
        else:
            # If there's only one token or none, use them directly
            processed_tokens = initial_tokens

        result = []
        for token in processed_tokens:
            if token == "<*>" and result and result[-1] == "<*>":
                continue
            result.append(token)

        processed_text = " ".join(result)
        processed_text = processed_text.replace("__DASH__", "-")
        doc = nlp(processed_text)
        tokens = []
        tokenPosTag = []
        for token in doc:
            if token.text.strip():
                if token.pos_ == "NUM" and self.has_numbers(token.text): # one will be <*> if no hasNumber()
                    tokens.append("<*>")
                else:
                    tokens.append(token.text)
            tokenPosTag.append(token.pos_)

        return tokens, tokenPosTag
