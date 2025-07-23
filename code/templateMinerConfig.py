# SPDX-License-Identifier: MIT

import ast
import configparser
import json
import logging
from typing import Collection, Optional

from drain3.masking import AbstractMaskingInstruction, MaskingInstruction

logger = logging.getLogger(__name__)


class TemplateMinerConfig:
    def __init__(self) -> None:
        self.engine = "Scope"
        self.profiling_enabled = False
        self.profiling_report_sec = 60
        self.snapshot_interval_minutes = 5
        self.snapshot_compress_state = True
        self.drain_extra_delimiters: Collection[str] = []
        self.drain_sim_th = 0.4
        self.drain_depth = 4
        self.drain_max_children = 100
        self.drain_max_clusters: Optional[int] = None
        self.masking_instructions: Collection[AbstractMaskingInstruction] = []
        self.mask_prefix = "<"
        self.mask_suffix = ">"
        self.parameter_extraction_cache_capacity = 3000
        self.parametrize_numeric_tokens = True
        self.bi_tree_support = False
        self.pool_support = False
        self.POS_support = False
        self.LLM_support = False

    def load(self, config_filename: str) -> None:
        parser = configparser.ConfigParser()
        read_files = parser.read(config_filename)
        if len(read_files) == 0:
            logger.warning(f"config file not found: {config_filename}")

        section_profiling = 'PROFILING'
        section_snapshot = 'SNAPSHOT'
        section_drain = 'CONFIGURATION'
        section_masking = 'MASKING'

        self.engine = parser.get(section_drain, 'engine', fallback=self.engine)

        self.profiling_enabled = parser.getboolean(section_profiling, 'enabled',
                                                   fallback=self.profiling_enabled)
        self.profiling_report_sec = parser.getint(section_profiling, 'report_sec',
                                                  fallback=self.profiling_report_sec)

        self.snapshot_interval_minutes = parser.getint(section_snapshot, 'snapshot_interval_minutes',
                                                       fallback=self.snapshot_interval_minutes)
        self.snapshot_compress_state = parser.getboolean(section_snapshot, 'compress_state',
                                                         fallback=self.snapshot_compress_state)

        drain_extra_delimiters_str = parser.get(section_drain, 'extra_delimiters',
                                                fallback=str(self.drain_extra_delimiters))
        self.drain_extra_delimiters = ast.literal_eval(drain_extra_delimiters_str)

        self.drain_sim_th = parser.getfloat(section_drain, 'sim_th',
                                            fallback=self.drain_sim_th)
        self.drain_depth = parser.getint(section_drain, 'depth',
                                         fallback=self.drain_depth)
        self.drain_max_children = parser.getint(section_drain, 'max_children',
                                                fallback=self.drain_max_children)
        self.drain_max_clusters = parser.getint(section_drain, 'max_clusters',
                                                fallback=self.drain_max_clusters)
        self.parametrize_numeric_tokens = parser.getboolean(section_drain, 'parametrize_numeric_tokens',
                                                            fallback=self.parametrize_numeric_tokens)
        self.bi_tree_support = parser.getboolean(section_drain, 'bi_tree_support',
                                         fallback=self.bi_tree_support)
        self.pool_support = parser.getboolean(section_drain, 'pool_support',
                                                       fallback=self.pool_support)
        self.POS_support = parser.getboolean(section_drain, 'POS_support',
                                         fallback=self.POS_support)
        self.LLM_support = parser.getboolean(section_drain, 'LLM_support',
                                         fallback=self.LLM_support)
        self.LLM_provider = parser.get(section_drain, 'LLM_provider', fallback='openai')
        self.LLM_model = parser.get(section_drain, 'LLM_model', fallback='gpt-3.5-turbo')
        self.LLM_api_key = parser.get(section_drain, 'LLM_api_key', fallback=None)
        self.LLM_thinking = parser.getboolean(section_drain, 'LLM_thinking',
                                              fallback=False)

        masking_instructions_str = parser.get(section_masking, 'masking',
                                              fallback=str(self.masking_instructions))
        self.mask_prefix = parser.get(section_masking, 'mask_prefix', fallback=self.mask_prefix)
        self.mask_suffix = parser.get(section_masking, 'mask_suffix', fallback=self.mask_suffix)
        self.parameter_extraction_cache_capacity = parser.getint(section_masking, 'parameter_extraction_cache_capacity',
                                                                 fallback=self.parameter_extraction_cache_capacity)

        masking_instructions = []
        masking_list = json.loads(masking_instructions_str)
        for mi in masking_list:
            instruction = MaskingInstruction(mi['regex_pattern'], mi['mask_with'])
            masking_instructions.append(instruction)
        self.masking_instructions = masking_instructions

