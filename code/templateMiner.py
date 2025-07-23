# SPDX-License-Identifier: MIT

import os
import sys
import base64
import logging
import re
import time
import zlib
from typing import Optional, Mapping, MutableMapping, NamedTuple, Sequence, Tuple, Union

import jsonpickle  # type: ignore[import]
from cachetools import LRUCache, cachedmethod

# Add the directory containing the module to sys.path
scope_dir = os.path.dirname(__file__)
sys.path.insert(0, scope_dir)  # Insert at the start of sys.path

from scope import Scope
from masking import LogMasker
from simpleProfiler import SimpleProfiler, NullProfiler, Profiler
from templateMinerConfig import TemplateMinerConfig


logger = logging.getLogger(__name__)

config_filename = 'scope.ini'

class TemplateMiner:

    def __init__(self, config: Optional[TemplateMinerConfig] = None):
        """
        Wrapper for SCOPE with persistence and masking support
        :param persistenceHandler: The type of persistence to use. When None, no persistence is applied.
        :param config: Configuration object. When none, configuration is loaded from default .ini file (if exist)
        """
        logger.info("Starting scope template miner")

        if config is None:
            logger.info(f"Loading configuration from {config_filename}")
            config = TemplateMinerConfig()
            config.load(config_filename)

        self.config = config
        self.profiler: Profiler = NullProfiler()

        if self.config.profiling_enabled:
            self.profiler = SimpleProfiler()

        param_str = f"{self.config.mask_prefix}*{self.config.mask_suffix}"

        self.scope = Scope(
            sim_th=self.config.drain_sim_th,
            depth=self.config.drain_depth,
            max_children=self.config.drain_max_children,
            max_clusters=self.config.drain_max_clusters,
            extra_delimiters=self.config.drain_extra_delimiters,
            profiler=self.profiler,
            param_str=param_str,
            parametrize_numeric_tokens=self.config.parametrize_numeric_tokens,
            bi_tree_support = self.config.bi_tree_support,
            pool_support = self.config.pool_support,
            POS_support = self.config.POS_support,
            LLM_support = self.config.LLM_support,
            LLM_provider=self.config.LLM_provider,
            LLM_model=self.config.LLM_model,
            LLM_api_key=self.config.LLM_api_key,
            LLM_thinking= self.config.LLM_thinking
        )

        self.masker = LogMasker(self.config.masking_instructions, self.config.mask_prefix, self.config.mask_suffix)
        self.parameter_extraction_cache: MutableMapping[Tuple[str, bool], str] = \
            LRUCache(self.config.parameter_extraction_cache_capacity)
        self.last_save_time = time.time()


    def add_log_message(self, log_message: str) -> Mapping[str, Union[str, int]]:
        self.profiler.start_section("total")

        logger.debug(f"Input log message: {log_message}")
        self.profiler.start_section("mask")
        masked_content = self.masker.mask(log_message)
        self.profiler.end_section()

        self.profiler.start_section("scope")
        logger.debug(f"Input log message with masked content: {masked_content}")
        cluster, change_type = self.scope.add_log_message(masked_content)
        self.profiler.end_section("scope")
        result: Mapping[str, Union[str, int]] = {
            "change_type": change_type,
            "cluster_id": cluster.templateId,
            "cluster_size": cluster.matchedLogSize,
            "template_mined": cluster.getTemplateStr(),
            "total_cluster_count": len(self.scope.clusters)
        }

        self.profiler.end_section("total")
        self.profiler.report(self.config.profiling_report_sec)
        return result

    @cachedmethod(lambda self: self.parameter_extraction_cache)
    def _get_template_parameter_extraction_regex(self,
                                                 log_template: str,
                                                 exact_matching: bool) -> Tuple[str, Mapping[str, str]]:
        param_group_name_to_mask_name = {}
        param_name_counter = [0]

        def get_next_param_name() -> str:
            param_group_name = f"p_{str(param_name_counter[0])}"
            param_name_counter[0] += 1
            return param_group_name

        # Create a named group with the respective patterns for the given mask-name.
        def create_capture_regex(_mask_name: str) -> str:
            allowed_patterns = []
            if exact_matching:
                # get all possible regex patterns from masking instructions that match this mask name
                masking_instructions = self.masker.instructions_by_mask_name(_mask_name)
                for mi in masking_instructions:
                    # MaskingInstruction may already contain named groups.
                    # We replace group names in those named groups, to avoid conflicts due to duplicate names.
                    if hasattr(mi, 'regex') and hasattr(mi, 'pattern'):
                        mi_groups = mi.regex.groupindex.keys()
                        pattern: str = mi.pattern
                    else:
                        # non regex masking instructions - support only non-exact matching
                        mi_groups = []
                        pattern = ".+?"

                    for group_name in mi_groups:
                        param_group_name = get_next_param_name()

                        def replace_captured_param_name(param_pattern: str) -> str:
                            _search_str = param_pattern.format(group_name)
                            _replace_str = param_pattern.format(param_group_name)
                            return pattern.replace(_search_str, _replace_str)

                        pattern = replace_captured_param_name("(?P={}")
                        pattern = replace_captured_param_name("(?P<{}>")

                    # support unnamed back-references in masks (simple cases only)
                    pattern = re.sub(r"\\(?!0)\d{1,2}", r"(?:.+?)", pattern)
                    allowed_patterns.append(pattern)

            if not exact_matching or _mask_name == "*":
                allowed_patterns.append(r".+?")

            # Give each capture group a unique name to avoid conflicts.
            param_group_name = get_next_param_name()
            param_group_name_to_mask_name[param_group_name] = _mask_name
            joined_patterns = "|".join(allowed_patterns)
            capture_regex = f"(?P<{param_group_name}>{joined_patterns})"
            return capture_regex

        # For every mask in the template, replace it with a named group of all
        # possible masking-patterns it could represent (in order).
        mask_names = set(self.masker.mask_names)

        # the Drain catch-all mask
        mask_names.add("*")

        escaped_prefix = re.escape(self.masker.mask_prefix)
        escaped_suffix = re.escape(self.masker.mask_suffix)
        template_regex = re.escape(log_template)

        # replace each mask name with a proper regex that captures it
        for mask_name in mask_names:
            search_str = escaped_prefix + re.escape(mask_name) + escaped_suffix
            while True:
                rep_str = create_capture_regex(mask_name)
                # Replace one-by-one to get a new param group name for each replacement.
                template_regex_new = template_regex.replace(search_str, rep_str, 1)
                # Break when all replaces for this mask are done.
                if template_regex_new == template_regex:
                    break
                template_regex = template_regex_new

        # match also messages with multiple spaces or other whitespace chars between tokens
        template_regex = re.sub(r"\\ ", r"\\s+", template_regex)
        template_regex = f"^{template_regex}$"
        return template_regex, param_group_name_to_mask_name

