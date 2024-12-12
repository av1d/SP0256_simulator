import cmudict
import numpy as np
import os
import re
import warnings
from scipy.io import wavfile

# a crude attempt at a GI SP0256-AL2 speech synthesis simulator
# https://github.com/av1d/

"""
MIT License

Copyright (c) 2024 av1d

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Note: the file '00_read_me.pdf' in /original_wav' claims the WAVs are
# under CC 3.0 license, but this is untrue as they're pirated. They are
# probably under copyright of General Instruments and not free at all.
# Therefore, only this software is covered with the above license.


cmu_dict = cmudict.dict()

# Location of allophone WAV files. In this case, ./modified_wav/
INPUT_WAV_PATH = os.path.join(os.getcwd(), "modified_wav") + "/"
# Where to output the output.wav file (current working dir)
OUTPUT_WAV_PATH = os.getcwd()
OUTPUT_FILE = "output.wav"

# True or False to ignore these punctuation events.
IGNORE_SPACES = True
IGNORE_PERIODS = True
IGNORE_COMMAS = True

# Terminal colors
GREEN = '\033[92m'
RESET = '\033[0m'

# All allophones which have multiple WAV files
numbered = {
    'BB': ['BB1.wav', 'BB2.wav'],
    'DD': ['DD1.wav', 'DD2.wav'],
    'DH': ['DH1.wav', 'DH2.wav'],
    'ER': ['ER1.wav', 'ER2.wav'],
    'GG': ['GG2.wav', 'GG3.wav'],
    'HH': ['HH1.wav', 'HH2.wav'],
    'KK': ['KK1.wav', 'KK2.wav', 'KK3.wav'],
    'NN': ['NN1.wav', 'NN2.wav'],
    'RR': ['RR1.wav', 'RR2.wav'],
    'TT': ['TT1.wav', 'TT2.wav'],
    'UW': ['UW1.wav', 'UW2.wav'],
    'YY': ['YY1.wav', 'YY2.wav']
}

# All allophones with single WAV files
unnumbered = {
    'AA': ['AA.wav'],
    'AE': ['AE.wav'],
    'AO': ['AO.wav'],
    'AR': ['AR.wav'],
    'AW': ['AW.wav'],
    'AX': ['AX.wav'],
    'AY': ['AY.wav'],
    'CH': ['CH.wav'],
    'EH': ['EH.wav'],
    'EL': ['EL.wav'],
    'EY': ['EY.wav'],
    'FF': ['FF.wav'],
    'GOT': ['GOT.wav'],
    'IH': ['IH.wav'],
    'IY': ['IY.wav'],
    'JH': ['JH.wav'],
    'LL': ['LL.wav'],
    'MM': ['MM.wav'],
    'NG': ['NG.wav'],
    'OR': ['OR.wav'],
    'OW': ['OW.wav'],
    'OY': ['OY.wav'],
    'PP': ['PP.wav'],
    'SH': ['SH.wav'],
    'SS': ['SS.wav'],
    'TH': ['TH.wav'],
    'UH': ['UH.wav'],
    'VV': ['VV.wav'],
    'WH': ['WH.wav'],
    'WW': ['WW.wav'],
    'XR': ['XR.wav'],
    'YR': ['YR.wav'],
    'ZH': ['ZH.wav'],
    'ZZ': ['ZZ.wav']
}


def warn_format(message, category, filename, lineno, line=None):
    return f"Warning: Skipping {filename} due to {message}.\n"

def text_to_allophones(text: str) -> dict:
    """
    Convert text to a dictionary of allophones using the
    CMU Pronunciation Dictionary. For more than 1 space,
    each space will be inserted into the list as a separate element.

    Args:
        text (str): The input text to be converted.

    Returns:
        dict: A dictionary where the keys are the words and the values are the corresponding allophones.

    Examples:
        >>> print( text_to_allophones("hello there") )
        >>> {'hello': ['HH', 'AH0', 'L', 'OW1'], 'there': ['DH', 'EH1', 'R']}
    """
    # Tokenize the text into words
    words = re.findall(r'\w+', text.lower())

    allophones_dict = {}

    for word in words:
        # Look up the pronunciation in the CMU dictionary
        pronunciations = cmu_dict.get(word, [])

        if pronunciations:
            # Use the first pronunciation if multiple are available
            pronunciation = pronunciations[0]
            allophones_dict[word] = pronunciation
        else:
            # If the word is not found, keep the original word
            allophones_dict[word] = [word]

    return allophones_dict

def allophone_contains_digit(allophone: str) -> list:
    """
    Checks if allophone contains digit.

    Args:
        allophone (str): The allophone string.

    Returns:
        list: A list containing the original allophone as element 0 and
              the modified allophone name without digits as element 1,
              if a digit is found in the string. Empty list if no digit
              is found in the string.
    """
    digits_removed = ''.join(
        char for char in allophone if not char.isdigit()
    )
    if digits_removed != allophone:
        return [allophone, digits_removed]
    return []

def find_closest_match(input_str: str) -> list:
    """
    Find the closest match. A single-letter input will try to match a
    double-letter allophone. Example: input "D" will first try to match
    "DD". If that isn't found, it will try to match the first letter of
    the allophone and return the first result it finds. Example: input
    of "D" will first return "DD" and ignore "DH".
    If still nothing is found, an empty list is returned.

    Args:
        str: allophone you are trying to match.

    Returns:
        list: if match found, a list containing allophone
        WAV file names. If no match found, an empty list.
    """
    for numbered_allophone in numbered:
        if len(input_str) == 1: # if input is only 1 char
            double_string = input_str + input_str # double it
            if numbered_allophone == double_string: # if it matches. Example: if input is Z and it matches ZZ
                #print(f"....Found double as the closest match: {numbered_allophone}.")
                return numbered[double_string] # return the WAV file name(s)
        elif len(input_str) > 1:
            single_letters_numbered = list(numbered_allophone) # split numbered allophone name into characters
            single_letters_input = list(input_str) # split the input into characters
            if single_letters_input[0] == single_letters_numbered[0]: # if the first characters match
                return numbered[numbered_allophone] # return the WAV file name(s)

    for unnumbered_allophone in unnumbered:
        if len(input_str) == 1: # if input is only 1 char
            double_string = input_str + input_str # double it
            if unnumbered_allophone == double_string: # if it matches. Example: if input is Z and it matches ZZ
                #print(f"....Found double as the closest match: {unnumbered_allophone}.")
                return unnumbered[double_string] # return the WAV file name(s)
        elif len(input_str) > 1:
            single_letters_unnumbered = list(unnumbered_allophone) # split unnumbered allophone name into characters
            single_letters_input = list(input_str) # split the input into characters
            if single_letters_input[0] == single_letters_unnumbered[0]: # if the first characters match
                return unnumbered[unnumbered_allophone] # return the WAV file name(s)

    #print("....No closest match found.")
    return [] # if we have failed to find a match, return empty list

def lookup_allophone(allophone: str) -> str:
    """
    Try to find a matching WAV file for the provided allophone.

    Args:
        str: string containing the allophone.

    Returns:
        str: string containing the allophone WAV file.
    """
    # Find punctiation.
    if allophone == 'SPACE':
        return 'SPACE.wav'
    elif allophone == 'PERIOD':
        return 'PERIOD.wav'
    elif allophone == 'COMMA':
        return 'COMMA.wav'

    # note: we're returning element 0 of the list for now because
    # we have no language rules for aspirated/unaspirated variations
    elif allophone in numbered: # if in numbered dict
        return numbered[allophone][0]
    elif allophone in unnumbered: # if in unnumbered dict
        return unnumbered[allophone][0]
    else: # if allophone not found in either dict
        allophone_has_digit = allophone_contains_digit(allophone) # see if it has a digit, if so, strip it
        if allophone_has_digit: # if it does
            modified_allophone = allophone_has_digit[1] # take the allophone name without digit
            if modified_allophone in numbered: # check if in numbered dict
                return numbered[modified_allophone][0]
            elif modified_allophone in unnumbered: # check if in unnumbered dict
                return unnumbered[modified_allophone][0]
            else:
                match = find_closest_match(allophone)
                if match:
                    return match[0]
                else:
                    return ""
        else: # if nothing found in any dictionary even after modifying the allophone name
            match = find_closest_match(allophone)
            if match:
                return match[0]
            else:
                return ""

def parse_input(input_text: str) -> list:
    """
    Breaks sentence and punctuation, including spaces,
    into individual elements.

    Args:
        str: original input text.

    Returns:
        list: list with parsed input text.

    Examples:
        >>> print( parse_input("hello there, alien.") )
        >>> ['hello', ' ', 'there', ',', ' ', 'alien', '.']
    """
    parsed = re.findall(r'\w+|\s|[^\w\s]', input_text)
    return parsed

def process_punctuation(parsed_input: str, allophone_dict: dict) -> list:
    """
    Converts a given text to a list of phonemes using a
    dictionary of word-to-phoneme mappings.

    Args:
        text (list): A list of words, punctuation, and spaces to beconverted to phonemes.
        phoneme_dict (dict): A dictionary mapping words to their corresponding phoneme lists.

    Returns:
        list: A list of phonemes representing the input text.

    Examples:
        >>> text = ['this', ',', ' ', 'is', ' ', 'a', ' ', 'test', '.']
        >>> phoneme_dict = {'this': ['DH', 'IH1', 'S'], 'is': ['IH1', 'Z'], 'a': ['AH0'], 'test': ['T', 'EH1', 'S', 'T']}
        >>> convert_to_phonemes(text, phoneme_dict)
        ['DH', 'IH1', 'S', 'COMMA', 'SPACE', 'IH1', 'Z', 'SPACE', 'AH0', 'SPACE', 'T', 'EH1', 'S', 'T', 'PERIOD']

    """

    c = []
    for word in parsed_input:
        if word in allophone_dict:
            c.extend(allophone_dict[word])
        elif word == ',':
            c.append('COMMA')
        elif word == ' ':
            c.append('SPACE')
        elif word == '.':
            c.append('PERIOD')
    return c

def prune_punctuation(allophones: list) -> list:
    """
    Prune punctuation from a list of allophones based on user-defined flags.

    Args:
        list: list of allophone and punctuation WAV files.

    Returns:
        list: list of updated allophone and punctuation WAV files.
    """
    result = allophones.copy()  # create a copy of the input list
    if IGNORE_SPACES:
        result = [item for item in result if item != 'SPACE.wav']
    if IGNORE_PERIODS:
        result = [item for item in result if item != 'PERIOD.wav']
    if IGNORE_COMMAS:
        result = [item for item in result if item != 'COMMA.wav']
    return result

def write_wav(wav_files: list):
    """
    Concatenates a list of WAV files in order starting at element 0
    into one single WAV file without gaps.

    Args:
        list: a list of WAV files.
    """
    combined_audio = []
    sample_rate = None
    for wav_file in wav_files:
        print(wav_file)
        current_sample_rate, audio_data = wavfile.read(
            os.path.join(
                INPUT_WAV_PATH,
                wav_file
            )
        )
        if sample_rate is None:
            sample_rate = current_sample_rate
        elif sample_rate != current_sample_rate:
            raise ValueError(
                "All WAV files must have the same sample rate."
            )
        combined_audio.extend(audio_data)


    wavfile.write(
        os.path.join(
            OUTPUT_WAV_PATH,
            OUTPUT_FILE
        ),
        sample_rate,
        np.array(combined_audio, dtype=np.int16)
    )
    return OUTPUT_FILE

def main():

    input_string = input("say> ") # the string you want converted
    parsed_input = parse_input(input_string) # returns list containing words/punctuation as separate elements
    allophone_dict = text_to_allophones(input_string) # returns dict
    processed_input = process_punctuation(parsed_input, allophone_dict) # final processed input text

    result = [] # the final allophone WAV file list
    for article in processed_input:
        wav_filename = lookup_allophone(article) # locate corresponding WAV file for the allophone
        if wav_filename:
            result.append(wav_filename)

    # remove corresponding WAV files if we're ignoring these punctuation types.
    if IGNORE_SPACES or IGNORE_PERIODS or IGNORE_COMMAS:
        result = prune_punctuation(result)

    # create the WAV file
    wav_filename = write_wav(result)

    label1 = f"{GREEN}Input string:{RESET}"
    label2 = f"{GREEN}Identified allophones:{RESET}"
    label3 = f"{GREEN}Corresponding WAV files:{RESET}"
    label4 = f"{GREEN}Output file:{RESET}"

    max_label_length = max(len(label1), len(label2), len(label3), len(label4))

    print(f"{label1:>{max_label_length}} {input_string}")
    print(f"{label2:>{max_label_length}} {allophone_dict}")
    print(f"{label3:>{max_label_length}} {', '.join(result)}")
    print(f"{label4:>{max_label_length}} {wav_filename}")


if __name__ == "__main__":
    main()

