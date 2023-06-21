from typing import List, Dict, Any
from python_heideltime import Heideltime
from bs4 import BeautifulSoup

def add_json_heideltime(text: str, heideltime_parser: Heideltime) -> List[Dict[str, Any]]:
    # Get Time Expressions
    xml_timex3 = heideltime_parser.parse(text)
    # Map the TIMEX Nodes into the Raw String Character Offsets
    soup  = BeautifulSoup(xml_timex3, 'xml')
    root = soup.find('TimeML')
    span_end = 0
    timex_all = []
    try:
        for timex in root.find_all('TIMEX3'):
            span_begin = span_end + root.text[span_end:].index(timex.text) - 1
            span_end = span_begin + len(timex.text)
            timex_dict = {'ID': timex.get('tid'), 'category': timex.get('type'), 'value': timex.get('value'), 'surfaceForm': timex.text, 'locationStart': span_begin, 'locationEnd': span_end, 'method': 'HeidelTime'}
            timex_all.append(timex_dict)
        return timex_all
    except:
        return []


if __name__ == "__main__":
    heideltime_parser = Heideltime()
    heideltime_parser.set_document_type('NEWS')
    print(heideltime_parser.parse('Yesterday, I bought a cat! It was born earlier this year.'))