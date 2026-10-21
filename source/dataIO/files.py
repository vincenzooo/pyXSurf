#from pprint import pprint
import numpy as np
import os

import warnings
warnings.filterwarnings("error", category=np.VisibleDeprecationWarning) 

def files_with_ext(base_folder, folder_glob_pattern= "*", file_extension=".*"):
    """
    Find files with a given extension within folders matching a glob pattern.

    :param base_folder: The base directory to start the search from.
    :param folder_glob_pattern: The glob pattern to match folders.
    :param file_extension: The extension of the files to find. Pass empty string for folders or files with no extension.
    :return: A list of paths to files matching the given extension in folders matching the glob pattern.
    
    N.B.: doesn't return files in the base_folder itself, but only in subfolders.
    """
    from pathlib import Path

    # Define the base path to start searching from
    base_path = Path(base_folder)

    # Initialize a list to hold the matched file paths
    matched_files = []

    # Find all folders matching the glob pattern
    for folder in base_path.glob(folder_glob_pattern):
        if folder.is_dir():  # Ensure the matched path is a directory
            # Use rglob to find all files with the given extension in the current folder and its subfolders
            matched_files.extend(str(f) for f in folder.rglob(f'*{file_extension}'))

    return matched_files

def read_blocks(filename,delimiter = '\n\n', comment = None,*args,**kwargs):
    """Read a file containing blocks of numerical data separated by a white line.
    
    Return a list, each element of which is a list of the lines in the block.  
    2022/06/29 converted result from dictionary to list.    
    """
    
    f = open(filename, 'r')
    content = f.read()
    f.close()
    pList=[]
    for block in content.split(delimiter):
        if (block.strip() != ''):
            """build a dic with a set of points, first line is used as key"""
            lines=block.split('\n')
            goodlines = [dd for dd in lines if (comment is None) or not(dd.startswith(comment))] 
            goodlines = [gl for gl in goodlines if len(gl.strip())!=0] #il caso di commento e' gia' escluso
            try:
                l = np.array([dd.split() for dd in goodlines],*args,**kwargs)
                l = np.array([dd.split() for dd in goodlines],*args,**kwargs)
            except np.VisibleDeprecationWarning:
                print ('Triying to convert splitted strings with unknown dtype results'+
                       ' in non corresponding lengths. Return array of strings.')
                l = goodlines
            except ValueError:
                print('Data type not corresponding to user-provided dtype. Return array of strings.')
                l = goodlines
                
            pList.append(np.array(l))

    return pList
    
def test_read_blocks(filename=None):
    
    def print_results(a, title):
        '''pass result and descriptive title for a test.'''
        
        print('\n====== ',title, ' ======\n')
        print('%i blocks read'%len(a))
        print('block shapes: ',[aa.shape for aa in a])
        print('block contents: ')
        for i,aa in enumerate(a):
            print('block #%i:\n'%(i+1),'[%s\n..\n..\n%s]\n'%(','.join(map(str,aa[:3])),','.join(map(str,aa[-2:]))))
        return a
    
    #infolder = r'C:\Users\kovor\Documents\python\pyXTel\source\pyProfile\test\input_data\'
    
    infolder = r'C:\Users\kovor\Documents\python\pyXTel\source\dataIO\test\test_data\blocks'
    
    if filename is None:
        filename = os.path.join(infolder,'data_blocks_spaced.dat')
    
    print('read "%s", contains three white-line-separated blocks with initial 3-line # comment followed by 94 couples of data. First block has 6-line comment.'%os.path.join(os.path.basename(os.path.dirname(filename)),os.path.basename(filename)))
    
    a = print_results(read_blocks(filename, comment = '#'),"comment = '#', ignore commented lines, lines are splitted. Return 2xN string array.") #array of strings 2 x N
    
    a = print_results(read_blocks(filename, comment = '#', dtype = float),"comment = '#' with float dtype, Return 2xN float array.") #array of floats 2 x N
    
    a = print_results(read_blocks(filename),"no comment or dtype, lines are splitted but they are inconsistent length. Return string vector with lines.")  # da warning in quanto lo split funziona creando liste. Se queste sono di lunghezze diversa,
                         # la conversione in array restituiscce un array monodimensionale (vettore) di oggetti (lista)

    a = print_results(read_blocks(filename, dtype = float),"dtype, but no comment character, comment lines break consistency. Return string vector with lines.") #lista di array
    
    
def count_header_lines(filename, comment_char="#", method = 'all',*args,**kwargs):
    """
    Count the number of header lines in a file, identified by a comment character.
    
    Args:
    - filename (str): The path to the text file.
    - comment_char (str): The character that indicates a header or comment line.
    - method (str): 'all' header starts from beginning and has comment_char in all lines
                    'last' consider last line starting with comment_char as the last line of header.
    
    Returns:
    - int: The number of header lines in the file.
    """
    
    with open(filename, 'r',*args,**kwargs) as file:
        lines = file.readlines()
        
    comment_lines = [l[0]== comment_char for l in lines]

    if method == 'last':
        header_lines = np.where(comment_lines)[0][-1]
    elif method == 'all':
        comment_indices = np.where(comment_lines)[0]
        header_lines = len(np.where(comment_indices == np.arange(len(comment_indices)))[0])
    else:
        raise ValueError("non valid method.")
    
    return header_lines
    
    
def head(fn,N=10):
    """return first n lines of file `fn`, without reading the other lines.
    
    from https://stackoverflow.com/questions/1767513/read-first-n-lines-of-a-file-in-python"""
    with open(fn) as myfile:
        return [next(myfile) for x in range(N)]
    

def search_encoding(filename,encode_list=None, verbose = False):
    """brute-force search for encoding.
    
    from stack-overflow, adding some extra function."""
    
    if encode_list is None:
        encode_list = ['ascii','big5','big5hkscs','cp037','cp273','cp424','cp437','cp500','cp720','cp737','cp775','cp850','cp852','cp855','cp856','cp857','cp858','cp860','cp861','cp862','cp863','cp864','cp865','cp866','cp869','cp874','cp875','cp932','cp949','cp950','cp1006','cp1026','cp1125','cp1140','cp1250','cp1251','cp1252','cp1253','cp1254','cp1255','cp1256','cp1257','cp1258','euc_jp','euc_jis_2004','euc_jisx0213','euc_kr','gb2312','gbk','gb18030','hz','iso2022_jp','iso2022_jp_1','iso2022_jp_2','iso2022_jp_2004','iso2022_jp_3','iso2022_jp_ext','iso2022_kr','latin_1','iso8859_2','iso8859_3','iso8859_4','iso8859_5','iso8859_6','iso8859_7','iso8859_8','iso8859_9','iso8859_10','iso8859_11','iso8859_13','iso8859_14','iso8859_15','iso8859_16','johab','koi8_r','koi8_t','koi8_u','kz1048','mac_cyrillic','mac_greek','mac_iceland','mac_latin2','mac_roman','mac_turkish','ptcp154','shift_jis','shift_jis_2004','shift_jisx0213','utf_32','utf_32_be','utf_32_le','utf_16','utf_16_be','utf_16_le','utf_7','utf_8','utf_8_sig']

    nfound = 0
    for encode in encode_list:
        try:
            df= pd.read_csv(fn, encoding = encode)
            print(encode)
            nfound = nfound + 1

        except Exception as e:
            if verbose:
                print(f"error: {e}")

    if nfound == 0:
        print('\n\nno encoding found.')
        if verbose:
            print('searched:','\n'.join(encode_list))
    else:
        print('\nfound %i encodings'%(nfound))

def program_path0():
    """first version. Both are from 
    https://stackoverflow.com/questions/51487645/get-path-of-script-containing-the-calling-function
    this one is not working on case 2 of test.
    See also:
    https://note.nkmk.me/en/python-script-file-path/#:~:text=In%20Python%2C%20you%20can%20get,python%20(or%20python3%20)%20command.
    """
    import inspect
    stack = inspect.stack()
    calling_context = next(context for context in stack if context.filename != __file__)
    return calling_context.filename
    #return stack


def program_path():
    """return the path
    
    to the calling file. """
    import sys

    namespace = sys._getframe(1).f_globals  # caller's globals
    #pprint(namespace)
    return namespace['__file__']


def test_program_path():
    """cases:
        1. called from input (VScode python file): 
            <ipython-input-11-0fddf82ac4ab>     
            
        if 2. called from ipython shell running this file:
            %run .../pyXTel/dataIO/files.py
            
        or 3. importing from this function from here with:
            from dataIO.files import test_program_path
            test_program_path()
        
        gives correct result:            
            caller's path: ...\pyXTel\dataIO\files.py
            

       if 4. called from a  .py file containing a copy of 
            this function.
            
            %run .../dataIO/test/test_program_path.py
            
            caller's path: ...\dataIO\test\test_program_path.py
            
        N.B. program_path0 gives incorrect results on case 2. only.
    """ 
    p = program_path0()  
    print("caller's path: ", p )
    return  p
    
    
        
if __name__ == "__main__":
    a = test_program_path()
    test_read_blocks()
    test_read_blocks(r'C:\Users\kovor\Documents\python\pyXTel\source\dataIO\test\test_data\blocks\data_blocks_spaced_irreg.dat')