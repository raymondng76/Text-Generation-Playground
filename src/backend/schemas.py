# -*- coding: utf-8 -*-

##################################################
## Configurations for FastAPI and transfomers Bloom model
##################################################
__author__ = "Raymond Ng"
__license__ = "MIT"
__maintainer__ = "Raymond Ng"
__version__ = "0.1.0"
__status__ = "Dev"
##################################################


from typing import List, Union

import pydantic


class Prompt(pydantic.BaseModel):
    """Prompt class with single text attribute"""

    text: str


class Prompts(pydantic.BaseModel):
    """Prompts class with list of prompts input and all text generation configuration"""

    prompts: List[Prompt]
    min_length: int
    max_length: int
    do_sample: bool
    early_stopping: bool
    num_beams: int
    temperature: float
    top_k: int
    top_p: float
    typical_p: float
    repetition_penalty: float
    length_penalty: float
    no_repeat_ngram_size: int
    max_time: Union[float, None]
    num_beam_groups: int
    diversity_penalty: float
    force_words: str
    bad_words: str
