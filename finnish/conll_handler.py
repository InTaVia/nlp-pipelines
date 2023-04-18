"""Handles label studio's .conll output file"""

from __future__ import annotations
from pathlib import Path
from string import punctuation

punctset = set(punctuation)


def read_conll(filename: str | Path) -> dict[str, list[tuple[str, str]]]:
    filename = Path(filename)
    sentences = {}
    sentence: list[str] = []
    sentence_parts: list[tuple[str, str]] = []
    for line in filename.open():
        line = line.strip()
        if not line:
            fixed_parts = _fix_conll_punctuation(sentence_parts)
            sentences[' '.join(sentence)] = fixed_parts
            sentence = []
            sentence_parts = []
            continue
        word, *_, label = line.split(' ')
        if word == '-DOCSTART-':
            continue
        sentence.append(word)
        sentence_parts.append((word, label))
    return sentences


def _fix_conll_punctuation(
    sentence_parts: list[tuple[str, str]], print_diff: bool = False
) -> list[tuple[str, str]]:
    new_parts = []
    last_indx = len(sentence_parts) - 1
    next_label = ''
    for indx, term in enumerate(sentence_parts):
        ttoken, tlabel = term
        # moving a label to the next token
        if next_label:
            new_parts.append((ttoken, next_label))
            next_label = ''
            continue
        if ttoken in punctset and tlabel != 'O':
            # first token in a named entity
            if indx == 0 or tlabel.startswith('B'):
                new_parts.append((ttoken, 'O'))
                next_label = tlabel
                continue
            # right parenthesis as continuation of a group
            if ttoken == ')' and _search_contiguous_punct(term, new_parts):
                new_parts.append(term)
                continue
            # last token overall
            if indx == last_indx:
                new_parts.append((ttoken, 'O'))
                continue
            nword, nlabel = sentence_parts[indx + 1]
            # last token in an entity or before punctuation
            if nlabel.startswith(('O', 'B')) or nword in ('.', ','):
                new_parts.append((ttoken, 'O'))
                continue
            # supposedly inside an entity
            new_parts.append(term)
        else:
            new_parts.append(term)
    if print_diff and new_parts != sentence_parts:
        for np, sp in zip(new_parts, sentence_parts):
            comp = sp[1] if sp[1] != np[1] else ''
            print(f"{np[0]:<20}: {np[1]} {'  <<' if comp else ''} {comp}")
        print('\n---\n')
    return new_parts


def _search_contiguous_punct(
    term: tuple[str, str], parts: list[tuple[str, str]], search_term: str = '('
) -> bool:
    wanted_label = f"I-{term[1].split('-')[1]}"
    for part in reversed(parts):
        if part[1] != wanted_label:
            return False
        if part[0] == search_term:
            return True
    return False
