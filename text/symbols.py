import yaml
from pathlib import Path


# Turkish (Türkçe)
_pad        = '_'
_punctuation = '!\'",():;.¿?…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
_special = '~•§©®™°¶†‡±×÷¢£€¥¤₺~•§©®™°¶†‡±'
_letters = 'ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZabcçdefgğhıijklmnnoöpqrsştuüvwxyz'
_numbers = '0123456789'

turkish_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_numbers)

# English (English)
_pad        = '_'
_punctuation = '!\'",():;.¿?…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
_special = '~•§©®™°¶†‡±×÷¢£€¥¤$~•§©®™°¶†‡±×÷¢£€¥¤$$†‡±×'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_numbers = '0123456789'

english_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_numbers)

# Spanish (Español)
_pad        = '_'
_punctuation = '!\'",():;.¿?¡…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
_special = '~•§©®™°¶†‡±×÷¢£€¥¤₧~•§©®™°¶†'
_letters = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyzÁÉÍÓÚÜáéíóúü'
_numbers = '0123456789'

spanish_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_numbers)

# French (Français)
_pad        = '_'
_punctuation = '!\'",():;.¿?¡…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
_special = '~°¶×÷£€¥¤¶'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ'
_numbers = '0123456789'

french_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_numbers)

# German (Deutsch)
_pad        = '_'
_punctuation = '!\'",():;.¿?¡…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
_special = '~•§©®™°¶†‡±×÷¢£€¥¤~•§©®™°¶†‡±×÷¢£€¥'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜßäöü'
_numbers = '0123456789'

german_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_numbers)

# Italian (Italiano)
_pad        = '_'
_punctuation = '!\'",():;.¿?¡…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
_special = '~•§©®™°¶†‡±×÷¢£€¥¤₤~•§©®™°¶†'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÈÉÌÒÓÙàèéìòóù'
_numbers = '0123456789'

italian_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_numbers)

# Portuguese (Português)
_pad        = '_'
_punctuation = '!\'",():;.¿?¡…—–«»„""''‚''‹›<>[]{}|/\\@#$%^&*+=~`'
_special = '~•§©®™°¶†‡±×÷¢£€¥¤'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÂÃÀÇÉÊÍÓÔÕÚáâãàçéêíóôõú'
_numbers = '0123456789'

portuguese_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_numbers)



config_path = Path("config/config.yaml")
with open(config_path, 'r', encoding='utf-8') as f: # Added encoding for safety
  try:
    config = yaml.safe_load(f)
  except Exception as e: # Catch other potential file reading errors
    symbols = turkish_symbols

if config:
  if str(config['language']) == "tr" or str(config['language']) == "turkish":
    symbols = turkish_symbols
  elif str(config['language']) == "en" or str(config['language']) == "english":
    symbols = english_symbols
  elif str(config['language']) == "es" or str(config['language']) == "spanish":
    symbols = spanish_symbols
  elif str(config['language']) == "fr" or str(config['language']) == "french":
    symbols = french_symbols
  elif str(config['language']) == "de" or str(config['language']) == "german":
    symbols = german_symbols
  elif str(config['language']) == "it" or str(config['language']) == "italian":
    symbols = italian_symbols
  elif str(config['language']) == "pt" or str(config['language']) == "portuguese":
    symbols = portuguese_symbols
  else:
    symbols = turkish_symbols
else:
  
  symbols = turkish_symbols
