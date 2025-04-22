import re
from num2words import num2words # type: ignore

# Basic cleaning functions
def lowercase(text):
    """Converts text to lowercase."""
    return text.lower()

def collapse_whitespace(text):
    """Converts multiple whitespaces to a single space and strips leading/trailing spaces."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_punctuation(text, language):
    """Removes punctuation marks based on the specified language."""
    if language == "turkish":
        punct_to_remove = '!\'",():;.¿?¡…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
    elif language == "arabic":
        punct_to_remove = '!\'",():;.¿?¡…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`،؛؟'
    else:
        punct_to_remove = '!\'",():;.¿?¡…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
    
    translator = str.maketrans('', '', punct_to_remove)
    return text.translate(translator)

def expand_numbers_with_num2words(text, language):
    """Converts numbers to words using the num2words library.
    
    Args:
        text (str): Text containing numbers to convert
        language (str): Language code for num2words (e.g., 'tr', 'en', 'fr')
        
    Returns:
        str: Text with numbers converted to words
    """
    # Language code mapping from our codes to num2words codes
    lang_map = {
        'turkish': 'tr',
        'english': 'en',
        'spanish': 'es',
        'french': 'fr',
        'german': 'de',
        'italian': 'it',
        'portuguese': 'pt',
        'russian': 'ru',
        'arabic': 'ar'
    }
    
    try:
      # Get num2words language code
      num2words_lang = lang_map.get(language, 'tr')
      
      # Find all numbers in the text
      # This regex matches integers and decimal numbers
      numbers = re.findall(r'\b\d+(\.\d+)?\b', text)
      
      # Sort found numbers by length (descending) to avoid partial replacements
      numbers_sorted = sorted(numbers, key=len, reverse=True)
      
      # Replace each number with its word equivalent
      for number in numbers_sorted:
          try:
              # Convert number to float or int as appropriate
              if '.' in number:
                  num_value = float(number)
              else:
                  num_value = int(number)
              
              # Get the word representation
              word = num2words(num_value, lang=num2words_lang)
              
              # Replace in text (with word boundaries to avoid partial replacements)
              text = re.sub(r'\b' + re.escape(number) + r'\b', word, text)
          except (ValueError, NotImplementedError):
              text = text # Original text
              continue
    except:
      text= text
      
    return text

# Language-specific functions
def normalize_turkish(text):
    """Special normalization for Turkish."""
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.replace('Ğ', 'ğ').replace('Ü', 'ü')
    text = text.replace('Ç', 'ç').replace('Ö', 'ö').replace('Ş', 'ş')
    return text
    
def normalize_spanish(text):
    """Special normalization for Spanish."""
    text = text.replace('Ñ', 'ñ')
    return text
    
def normalize_french(text):
    """Special normalization for French."""
    text = text.replace('Ç', 'ç').replace('Œ', 'œ').replace('Æ', 'æ')
    return text
    
def normalize_german(text):
    """Special normalization for German."""
    text = text.replace('ẞ', 'ß')  # Capital ß character
    return text
    
def normalize_arabic(text):
    """Special normalization for Arabic."""
    return text


def basic_cleaners(text):
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

def turkish_cleaners(text):
  text = normalize_turkish(text)
  text = basic_cleaners(text)
  text = expand_numbers_with_num2words(text, 'turkish')
  text = remove_punctuation(text, 'turkish')
  return text

def english_cleaners(text):
  text = basic_cleaners(text)
  text = expand_numbers_with_num2words(text, 'english')
  text = remove_punctuation(text, 'english')
  return text

def spanish_cleaners(text):
  text = normalize_spanish(text)
  text = basic_cleaners(text)
  text = expand_numbers_with_num2words(text, 'spanish')
  text = remove_punctuation(text, 'spanish')
  return text

def french_cleaners(text):
  text = normalize_french(text)
  text = basic_cleaners(text)
  text = expand_numbers_with_num2words(text, 'french')
  text = remove_punctuation(text, 'french')
  return text

def german_cleaners(text):
  text = normalize_german(text)
  text = basic_cleaners(text)
  text = expand_numbers_with_num2words(text, 'german')
  text = remove_punctuation(text, 'german')
  return text

def italian_cleaners(text):
  text = basic_cleaners(text)
  text = expand_numbers_with_num2words(text, 'italian')
  text = remove_punctuation(text, 'italian')
  return text

def portuguese_cleaners(text):
  text = basic_cleaners(text)
  text = expand_numbers_with_num2words(text, 'portuguese')
  text = remove_punctuation(text, 'portuguese')
  return text



