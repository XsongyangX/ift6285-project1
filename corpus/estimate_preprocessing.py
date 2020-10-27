from typing import Callable, Dict, List, Tuple

def reductive_mapping(voc: Dict[str, int], mapping: Callable[[str], str]) -> Tuple[Dict[str, int], int, int]:
    """Estimates how the vocabulary will change after this reductive mapping

    Args:
        voc (Dict[str, int]): Initial vocabulary
        mapping (Callable[[str], str]): Reductive mapping

    Returns:
        Dict[str, int]: Mapped vocabulary
        int: Change in type counts
        int: Number of tokens affected
    """
    preprocessed_voc : Dict[str, int] = {}
    tokens_affected = 0
    for key, val in voc.items():
        lowered = mapping(key)
        if lowered in preprocessed_voc: 
                tokens_affected += val
                preprocessed_voc[lowered] += val
        else:
                preprocessed_voc[lowered] = val

    return preprocessed_voc, len(voc) - len(preprocessed_voc), tokens_affected

def expansive_mapping(voc: Dict[str, int], mapping: Callable[[str], List[str]]):
    preprocessed_voc : Dict[str, int] = {}
    tokens_affected = 0

    def try_to_add_to_voc(voc: Dict[str, int], key: str, val: int):
        if key in voc:
            voc[key] += val
        else:
            voc[key] = val

    for key, val in voc.items():
        separated = mapping(key)
        if len(separated) != 1:
            tokens_affected += val
        for separated_fragment in separated:
            try_to_add_to_voc(preprocessed_voc, separated_fragment, val)
        

    return preprocessed_voc, len(voc) - len(preprocessed_voc), tokens_affected