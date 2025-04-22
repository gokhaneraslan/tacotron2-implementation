from text import cleaners
from text.symbols import symbols


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
  """
    Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    
    Args:
        text (str): The text to convert to a sequence
        cleaner_specs (list): List of tuples with (language, level) for cleaning the text
                              Example: [('english'), ('turkish')]
    
    Returns:
        list: A sequence of symbol IDs corresponding to the text
  """
  cleaned_text = _clean_text(text, cleaner_names)
  sequence = _symbols_to_sequence(cleaned_text)

  return sequence


def sequence_to_text(sequence):
  """
    Converts a sequence of IDs back to a string of text.
    
    Args:
        sequence (list): A list of integer IDs corresponding to symbols
        
    Returns:
        str: The decoded text
  """
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception(f'Unknown cleaner! {name}')
    text = cleaner(text)
  return text


def _symbols_to_sequence(text_symbols):
  """
    Converts a string of symbols to a sequence of symbol IDs.
    
    Args:
        text_symbols (str): String of symbols to convert
        
    Returns:
        list: List of symbol IDs
  """
  return [_symbol_to_id[s] for s in text_symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
  """
    Determines if a symbol should be kept in the sequence.
    
    Args:
        s (str): The symbol to check
        
    Returns:
        bool: True if the symbol should be kept, False otherwise
  """
  return s in _symbol_to_id and s != '_' and s != '~'