# src/utils/number_utils.py
import re
from num2words import num2words


def convert_numbers_to_words(text: str, lang: str = 'en') -> str:
    """
    Caută numere întregi în text și le înlocuiește cu cuvinte.
    Ex: "125" -> "o sută douăzeci și cinci" (pentru ro)
    Ex: "125" -> "one hundred twenty-five" (pentru en)
    
    Args:
        text: Textul care conține numere
        lang: Codul limbii ('ro', 'ro-RO', 'en', 'en-US', etc.)
    
    Returns:
        Textul cu numerele convertite în cuvinte
    """
    if not text:
        return text

    # Detectăm limba (ro sau en)
    # Codurile pot veni ca 'ro', 'ro-RO', 'en', 'en-US'
    lang_code = 'ro' if lang and lang.strip().lower().startswith('ro') else 'en'
    
    def replace_match(match):
        num_str = match.group()
        try:
            # Convertim șirul numeric în număr
            number = int(num_str)
            # Obținem textul (ex: one hundred twenty-five)
            return num2words(number, lang=lang_code)
        except Exception:
            # Fallback: returnăm numărul original dacă apare o eroare
            return num_str

    # Regex simplu pentru secvențe de cifre (\d+)
    # Va transforma "100" -> "o sută", "25 mere" -> "douăzeci și cinci mere"
    return re.sub(r'\d+', replace_match, text)
